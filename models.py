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
CLASSES = ["benign", "malignant", "normal"]  
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_resnet(path="weights/efficientnet_ultrasound.pth"):
    m = tv_models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

def predict_image(model, pil_img):
    x = _tf(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0].tolist()
    ui_classes = ["Benign", "Malignant", "Normal"]
    return dict(zip(ui_classes, p))

# ---------- 2) Numeric SVM Classifier ----------
FEATURES = [
    "texture_mean", "concave points_mean", "radius_se", "area_se", 
    "compactness_se", "radius_worst", "texture_worst", 
    "area_worst", "concavity_worst", "concave points_worst"
]

def load_values_model(path="weights/values_model.pkl"):
    return joblib.load(path)

def predict_values(model, values: dict):
    X = np.array([[values[f] for f in FEATURES]], dtype=float)
    proba = model.predict_proba(X)[0]
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
_init_error = None  

def _init_rag():
    global _embed_model, _faiss_index, _subset_df, _rerank_model, _qwen_tokenizer, _qwen_model, _init_error
    if _qwen_model is not None or _init_error is not None:
        return  

    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import faiss
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import hf_hub_download
        import pickle

        print("Downloading precomputed FAISS index and data from Hugging Face...")
        index_path = hf_hub_download(repo_id="maramyoussef0/medical-rag-data", filename="medical_faiss.index", repo_type="dataset")
        df_path = hf_hub_download(repo_id="maramyoussef0/medical-rag-data", filename="subset_df.pkl", repo_type="dataset")

        # Load instantly without heavy computation!
        _faiss_index = faiss.read_index(index_path)
        with open(df_path, "rb") as f:
            _subset_df = pickle.load(f)

        print("Loading embedding & reranking models...")
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        _rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print("Loading Qwen language model locally...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"  
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _qwen_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
        
    except Exception as e:
        _init_error = str(e)
        print(f"RAG initialization failed: {_init_error}")
