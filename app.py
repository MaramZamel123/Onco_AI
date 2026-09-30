import re
import streamlit as st
st.set_page_config(page_title="Dr. Nana – Breast Health Assistant", page_icon="🩺", layout="wide")
from PIL import Image
from doctor_component import render_doctor
import models as M
import os

FRIENDLY = {  
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
        try: 
            out[k] = fn()
        except Exception as e: 
            out[k] = None
            st.sidebar.warning(f"{k} model not loaded: {e}")
            
    try:
        M._init_rag()
    except Exception as e:
        st.sidebar.warning(f"RAG pre-load warning: {e}")
        
    return out

models = load_all()

ss = st.session_state
ss.setdefault("msgs", [("assistant", "Hello, I'm Dr. Nana. Ask me a health question, upload a scan, or enter your lab values.")])
ss.setdefault("mood", "wave")
ss.setdefault("say", "Hello, I'm Dr. Nana")
ss.setdefault("n", 0)
ss.setdefault("scan", None)
ss.setdefault("last_file", None)
ss.setdefault("pred_vals", None)
ss.setdefault("theme", "Clinical")
ss.setdefault("active_tab", "Chat")

def set_mood(m, say): 
    ss.mood, ss.say, ss.n = m, say, ss.n + 1

def react(probs):
    label = max(probs, key=probs.get)
    conf = probs[label] * 100
    if conf < 60: 
        set_mood("think", "I'm not fully certain. A specialist review is advised.")
        return
    if label == "Malignant": 
        set_mood("worry", "This needs a specialist's attention immediately.")
    elif label == "Normal": 
        set_mood("good", "Great news! Everything looks normal.")
    else: 
        set_mood("reassure", "It appears benign, but let's keep monitoring it closely.")

def bars(probs):
    for k, v in probs.items(): 
        st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v*100:.1f}%")

# ---- Top Header Bar with Theme Switcher ----
col_title, col_themes = st.columns([3, 2])
with col_title:
    st.markdown("### 🩺 Dr. Nana <span style='font-size:16px; font-weight:normal; color:gray;'>Breast Ultrasound & Health Assistant</span>", unsafe_allow_html=True)
with col_themes:
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        if st.button("Clinical", use_container_width=True): ss.theme = "Clinical"
    with t_col2:
        if st.button("Midnight", use_container_width=True): ss.theme = "Midnight"
    with t_col3:
        if st.button("Warm", use_container_width=True): ss.theme = "Warm"

st.divider()

# ---- Main Layout: Left Doctor Card Column, Right Panel Column ----
left, right = st.columns([1, 2.2], gap="large")
doc_slot = left.empty()

with right:
    # ---- Custom Tab Buttons Header ----
    tab_col1, tab_col2, tab_col3 = st.columns(3)
    with tab_col1:
        if st.button("Chat", use_container_width=True, type="primary" if ss.active_tab == "Chat" else "secondary"):
            ss.active_tab = "Chat"
            set_mood("happy", "How can I help?")
            st.rerun()
    with tab_col2:
        if st.button("Scan", use_container_width=True, type="primary" if ss.active_tab == "Scan" else "secondary"):
            ss.active_tab = "Scan"
            set_mood("listen", "Upload your breast ultrasound image.")
            st.rerun()
    with tab_col3:
        if st.button("Values", use_container_width=True, type="primary" if ss.active_tab == "Values" else "secondary"):
            ss.active_tab = "Values"
            set_mood("listen", "Enter the values from your report.")
            st.rerun()

    prompt = None

    if ss.active_tab == "Chat":
        st.markdown("---")
        for role, txt in ss.msgs: 
            st.chat_message(role).write(txt)

        # Quick Action Chips
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("Early signs?"):
                prompt = "What are early signs of breast cancer?"
        with qc2:
            if st.button("Screening?"):
                prompt = "How often should I get screened?"
        with qc3:
            if st.button("Upload scan"):
                prompt = "Upload scan"

        # Form Input Box for Typing Messages
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            with col_input:
                user_text = st.text_input("Ask Dr. Nana a health question...", label_visibility="collapsed", placeholder="Ask Dr. Nana a health question…")
            with col_send:
                submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

        if submitted and user_text:
            prompt = user_text

    elif ss.active_tab == "Scan":
        st.markdown("---")
        st.markdown("Upload your breast ultrasound image (JPG / PNG, ultrasound images only).")
        f = st.file_uploader("Upload breast ultrasound image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if f is None: 
            ss.last_file = None
        else:
            img = Image.open(f)
            st.image(img, width=300)
            if models["resnet"] is None: 
                st.error("Ultrasound weights not found (weights/efficientnet_ultrasound.pth).")
            else:
                if ss.last_file != (f.name, f.size):
                    with st.spinner("Analyzing image…"): 
                        ss.scan = M.predict_image(models["resnet"], img)
                    ss.last_file = (f.name, f.size)
                    react(ss.scan)
                bars(ss.scan)

    elif ss.active_tab == "Values":
        st.markdown("---")
        st.markdown("Enter your lab measurements from the report:")
        with st.form("lab_form"):
            c = st.columns(2)
            pred_vals = {}
            for i, (k, (lab, hlp, ex)) in enumerate(FRIENDLY.items()):
                pred_vals[k] = c[i % 2].number_input(lab, value=ex, min_value=0.0, format="%.4f", help=hlp)
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                go = st.form_submit_button("Analyze values", type="primary", use_container_width=True)
            with f_col2:
                fill_ex = st.form_submit_button("Fill example", use_container_width=True)

        if go or fill_ex:
            if models["values"] is None: 
                st.error("Model not found (weights/values_model.pkl).")
            else: 
                ss.pred_vals = M.predict_values(models["values"], pred_vals)
                react(ss.pred_vals)
        if ss.pred_vals: 
            bars(ss.pred_vals)

    st.caption("AI-assisted screening support for educational use. It is not a medical diagnosis; please consult a qualified physician.")

# ---- Handle Chat Prompt Processing ----
if prompt:
    ss.msgs.append(("user", prompt))
    low = prompt.lower()
    
    if re.match(r"(hi|hello|hey|salam|good (morning|evening))", low): 
        reply, m, say = "Hello! How can I help you today?", "wave", "Hello! Nice to see you."
    elif "thank" in low: 
        reply, m, say = "You're very welcome! Take care of yourself.", "good", "Happy to help!"
    elif re.search(r"scared|afraid|worried|anxious|fear|nervous", low):
        reply, m, say = "It's completely natural to feel that way. I'm here with you, and we can go through this step by step.", "reassure", "I'm here with you."
    elif re.search(r"scan|image|upload", low):
        reply, m, say = "Sure! Upload your scan in the Scan tab", "happy", "Send me the picture!"
        ss.active_tab = "Scan"
    elif re.search(r"value|measure|number|lab", low):
        reply, m, say = "Let's enter your report values", "think", "Fill in the values."
        ss.active_tab = "Values"
    else:
        with st.spinner("Dr. Nana is thinking…"): 
            reply = M.rag_answer(prompt)
        m, say = "talk", reply[:60] + "…"
        
    ss.msgs.append(("assistant", reply))
    set_mood(m, say)
    st.rerun()

# ---- Left Column: Doctor Card & Mood Preview Buttons ----
with doc_slot.container():
    st.info(f"**{ss.say}**")
    st.text(f"mood: {ss.mood}")
    st.markdown("Preview moods:")
    
    m_cols = st.columns(4)
    moods_list = ["wave", "happy", "think", "worry", "good", "talk", "listen", "reassure", "wow", "sleepy"]
    for idx, m_name in enumerate(moods_list):
        with m_cols[idx % 4]:
            if st.button(m_name, key=f"mood_{m_name}", use_container_width=True):
                set_mood(m_name, f"Feeling {m_name}!")
                st.rerun()
                
    render_doctor(ss.mood, ss.say, ss.n)
