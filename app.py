import re
import streamlit as st
from PIL import Image
from doctor_component import render_doctor
import models as M
import os
import urllib.request
import streamlit as st


@st.cache_resource
def download_rag_data():
    with st.spinner("Downloading required RAG data files from cloud storage... Please wait."):
        if not os.path.exists("train.json"):
            train_url = "https://drive.google.com/uc?export=download&id=1P_0eMdFCMUmjMRF6JkOW7YqsdFn8sYXQ"
            urllib.request.urlretrieve(train_url, "train.json")
            
        if not os.path.exists("validation.json"):
            val_url = "https://drive.google.com/uc?export=download&id=1FlAnM0lhp4CrdhQioB2RXhus1KRndKp6"
            urllib.request.urlretrieve(val_url, "validation.json")

download_rag_data()


st.title("My Medical AI App")

st.set_page_config(page_title="Dr. Nana", page_icon="🩺", layout="wide")

FRIENDLY = {  # original name: (label, help, example value)
    "texture_mean": ("Average Tissue Texture", "Pixel variation in tissue image", 19.0),
    "concave points_mean": ("Average Indentations on Cell Edges", "Count of small dents", 0.05),
    "radius_se": ("Cell Size Variability", "Spread of cell radius", 0.4),
    "area_se": ("Cell Area Variability", "Spread of cell area", 40.0),
    "compactness_se": ("Cell Shape Tightness Variability", "Spread of compactness", 0.025),
    "radius_worst": ("Largest Cell Radius", "Biggest cell measured", 16.0),
    "texture_worst": ("Roughest Tissue Texture", "Highest texture value", 25.0),
    "area_worst": ("Largest Cell Area", "Biggest cell area", 880.0),
    "concavity_worst": ("Deepest Cell Edge Dent", "Worst dent depth", 0.27),
    "concave points_worst": ("Most Indentations on a Cell Edge", "Worst dent count", 0.11),
}

@st.cache_resource
def load_all():
    out = {}
    for k, fn in (("resnet", M.load_resnet), ("values", M.load_values_model)):
        try: out[k] = fn()
        except Exception as e: out[k] = None; st.sidebar.warning(f"{k} model not loaded: {e}")
    return out
models = load_all()

ss = st.session_state
ss.setdefault("msgs", [("assistant", "Hello, I'm Dr. Nana. Ask me a health question, upload an ultrasound image, or enter your lab values.")])
ss.setdefault("mood", "wave"); ss.setdefault("say", "Hello, I'm Dr. Nana"); ss.setdefault("n", 0)
ss.setdefault("scan", None); ss.setdefault("last_file", None); ss.setdefault("pred_vals", None)

def set_mood(m, say): ss.mood, ss.say, ss.n = m, say, ss.n + 1

def react(probs):
    label = max(probs, key=probs.get); conf = probs[label] * 100
    if conf < 60: set_mood("think", "I'm not fully certain. A specialist review is advised."); return
    if label == "Malignant": set_mood("worry", "This needs a specialist's attention.")
    elif label == "Normal": set_mood("good", "Good news!")
    else: set_mood("reassure", "Benign, but let's keep watch.")

def bars(probs):
    for k, v in probs.items(): st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v*100:.1f}%")

# ---- chat input (handled before drawing so the doctor reacts immediately) ----
prompt = st.chat_input("Ask Dr. Nana a health question…")
if prompt:
    ss.msgs.append(("user", prompt)); low = prompt.lower()
    if re.match(r"(hi|hello|hey|salam)", low): reply, m, say = "Hello! How can I help you today?", "wave", "Hello! Nice to see you."
    elif "thank" in low: reply, m, say = "You're welcome! Take care of yourself.", "good", "Happy to help!"
    elif re.search(r"scared|afraid|worried|anxious|fear|nervous", low):
        reply, m, say = "It's completely natural to feel that way. I'm here with you, step by step.", "reassure", "I'm here with you."
    else:
        with st.spinner("Dr. Nana is thinking…"): reply = M.rag_answer(prompt)
        m, say = "talk", reply[:70] + "…"
    ss.msgs.append(("assistant", reply)); set_mood(m, say)

left, right = st.columns([1, 2.2], gap="large")
doc_slot = left.empty()

with right:
    st.markdown("### Dr. Nana · Breast Ultrasound & Health Assistant")
    t_chat, t_scan, t_vals = st.tabs(["Chat", "Ultrasound scan", "Lab values"])

    with t_chat:
        for role, txt in ss.msgs: st.chat_message(role).write(txt)

    with t_scan:
        f = st.file_uploader("Upload breast ultrasound image (JPG / PNG, ultrasound only)", type=["jpg", "jpeg", "png"])
        if f is None: ss.last_file = None
        else:
            img = Image.open(f); st.image(img, width=300)
            if models["resnet"] is None: st.error("Ultrasound weights not found (weights/efficientnet_ultrasound.pth).")
            else:
                if ss.last_file != (f.name, f.size):
                    with st.spinner("Analyzing image…"): ss.scan = M.predict_image(models["resnet"], img)
                    ss.last_file = (f.name, f.size); react(ss.scan)
                bars(ss.scan)

    with t_vals:
        with st.form("lab_form"):
            c = st.columns(2); pred_vals = {}
            for i, (k, (lab, hlp, ex)) in enumerate(FRIENDLY.items()):
                pred_vals[k] = c[i % 2].number_input(lab, value=ex, min_value=0.0, format="%.4f", help=hlp)
            go = st.form_submit_button("Analyze values", type="primary")
        if go:
            if models["values"] is None: st.error("Model not found (weights/values_model.pkl).")
            else: ss.pred_vals = M.predict_values(models["values"], pred_vals); react(ss.pred_vals)
        if ss.pred_vals: bars(ss.pred_vals)

    st.caption("AI-assisted screening support for educational use. Not a medical diagnosis; please consult a qualified physician.")

with doc_slot.container(): render_doctor(ss.mood, ss.say, ss.n)
