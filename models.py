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
import pickle
import streamlit as st

# ---------- Helper to get HF token safely ----------
def _get_token():
    return os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")

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

# ---------- 3) Healthcare RAG Chatbot (Optimized for Speed) ----------
_embed_model = None
_faiss_index = None
_subset_df = None
_init_error = None  

def rag_answer(query: str) -> str:
    _init_rag()
    if _faiss_index is None:
        return f"RAG system failed to load. Details: {_init_error}"

    try:
        from huggingface_hub import InferenceClient
        import time

        # 1. Retrieve context from FAISS instantly
        query_vector = _embed_model.encode([query], convert_to_numpy=True)
        _, indices = _faiss_index.search(query_vector, 3)
        
        retrieved_context = ""
        for i, idx in enumerate(indices[0]):
            doc_text = _subset_df.iloc[idx]['combined_doc']
            retrieved_context += f"--- Reference Case {i+1} ---\n{doc_text}\n\n"

        system_prompt = (
            "You are an empathetic and knowledgeable medical AI assistant. "
            "Answer the patient's question accurately using ONLY the provided reference cases from medical history below. "
            "If the answer cannot be found in the references, state so cautiously."
        )
        user_content = f"### Reference Cases:\n{retrieved_context}\n### Patient Query:\n{query}\n### Doctor's Professional Response:"

        token = _get_token()
        
        # Using a widely supported router model with the :cheapest tag for serverless access
        client = InferenceClient(token=token)
        
        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct:cheapest",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                break
            except Exception as api_err:
                if "503" in str(api_err) or "loading" in str(api_err).lower():
                    print(f"Model is waking up (cold start), retrying in 5 seconds... (Attempt {attempt+1}/3)")
                    time.sleep(5)
                else:
                    raise api_err

        if response is None:
            return "The AI model container is taking too long to wake up from sleep mode. Please try asking your question again in a moment."
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"
