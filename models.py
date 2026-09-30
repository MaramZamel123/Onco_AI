"""Loaders + predict functions adjusted for your notebooks."""
import os
import joblib
import torch
import torch.nn as nn
from torchvision import models as tv_models, transforms
from PIL import Image
import json
import pandas as pd
import numpy as np

# ---------- 1) EfficientNet Ultrasound Classifier ----------
CLASSES = ["benign", "malignant", "normal"]  # Matches your BUSI dataset folders in cancer2.py
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_resnet(path="weights/efficientnet_ultrasound.pth"):
    # You saved efficientnet_b0 in cancer2.py
    m = tv_models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

def predict_image(model, pil_img):
    x = _tf(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0].tolist()
    # Map back to capitalized labels for the UI
    ui_classes = ["Benign", "Malignant", "Normal"]
    return dict(zip(ui_classes, p))

# ---------- 2) Numeric SVM Classifier ----------
# The 10 features selected by RFE in your cancer1.py notebook
FEATURES = [
    "texture_mean", "concave points_mean", "radius_se", "area_se", 
    "compactness_se", "radius_worst", "texture_worst", 
    "area_worst", "concavity_worst", "concave points_worst"
]

def load_values_model(path="weights/values_model.pkl"):
    return joblib.load(path)  # Loads your saved SVM model

def predict_values(model, values: dict):
    X = np.array([[values[f] for f in FEATURES]], dtype=float)
    proba = model.predict_proba(X)[0]
    # Classes are typically 0 (Benign) and 1 (Malignant)
    classes = list(model.classes_)
    mal_idx = classes.index(1) if 1 in classes else 1
    mal_prob = proba[mal_idx]
    return {"Benign": float(1 - mal_prob), "Malignant": float(mal_prob)}

# ---------- 3) Healthcare RAG Chatbot (FAISS + Qwen) ----------
_embed_model = None
_faiss_index = None
_subset_df = None
_rerank_model = None
_qwen_tokenizer = None
_qwen_model = None
_init_error = None  # To track exact errors if any occur

def _init_rag():
    global _embed_model, _faiss_index, _subset_df, _rerank_model, _qwen_tokenizer, _qwen_model, _init_error
    if _qwen_model is not None or _init_error is not None:
        return  # Already attempted / loaded

    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import faiss
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import hf_hub_download

        # Automatically download and cache the actual JSON file (bypassing LFS pointer issues)
        print("Downloading train.json from Hugging Face...")
        train_file_path = hf_hub_download(
            repo_id="maramyoussef0/medical-rag-data",
            filename="train.json",
            repo_type="dataset"
        )

        # Load dataset safely
        train_df = pd.DataFrame(json.load(open(train_file_path, "r", encoding="utf-8")))
        if 'patient_message' in train_df.columns and 'doctor_response' in train_df.columns:
            train_df['combined_doc'] = "Patient: " + train_df['patient_message'].astype(str) + " \nDoctor Response: " + train_df['doctor_response'].astype(str)
        
        # REDUCED to 2,000 rows so it fits safely inside Streamlit Cloud's free RAM limit
        _subset_df = train_df.head(2000).copy()

        print("Loading embedding & reranking models for RAG...")
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        _rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Build FAISS index
        documents = _subset_df['combined_doc'].tolist()
        embeddings = _embed_model.encode(documents, batch_size=32, convert_to_numpy=True)
        _faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        _faiss_index.add(embeddings)

        print("Loading Qwen language model locally...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"  
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu"  
        )
    except Exception as e:
        _init_error = str(e)
        print(f"RAG initialization failed: {_init_error}")
