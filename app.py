import streamlit as st
import models as M
import re
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Dr. Nana – Breast Health Assistant", 
    page_icon="🩺", 
    layout="wide"
)

# Initialize RAG and models
try:
    M._init_rag()
except Exception as e:
    st.sidebar.warning(f"RAG warning: {e}")

@st.cache_resource
def load_all_models():
    return {
        "resnet": getattr(M, "load_resnet", lambda: None)(),
        "values": getattr(M, "load_values_model", lambda: None)()
    }

models = load_all_models()

# Session State Setup
ss = st.session_state
ss.setdefault("msgs", [("assistant", "Hello, I'm Dr. Nana. Ask me a health question, upload a scan, or enter your lab values.")])
ss.setdefault("mood", "wave")
ss.setdefault("say", "Hello, I'm Dr. Nana")
ss.setdefault("active_tab", "Chat")

# 2. Custom CSS to match your clean card styling
st.markdown("""
<style>
.stApp { background-color: #f1f6f8; }
.card { background: #fff; border-radius: 14px; padding: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }
.bub { background: #e2edf1; border-radius: 16px; padding: 12px 16px; font-size: 14px; margin-bottom: 8px; font-weight: 500; color: #1f2f3a; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("### 🩺 Dr. Nana <span style='font-size:13px; font-weight:normal; color:gray;'>Breast Ultrasound & Health Assistant</span>", unsafe_allow_html=True)
st.divider()

# Layout Columns (Left: Doctor Card, Right: Panels)
left_col, right_col = st.columns([1, 2.2], gap="large")

with left_col:
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <div class="bub">{ss.say}</div>
        <div style="font-size: 12px; color: gray;">mood: {ss.mood}</div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    # Navigation Tabs
    t1, t2, t3 = st.columns(3)
    with t1:
        if st.button("Chat", use_container_width=True, type="primary" if ss.active_tab == "Chat" else "secondary"):
            ss.active_tab = "Chat"
            ss.mood, ss.say = "happy", "How can I help?"
            st.rerun()
    with t2:
        if st.button("Scan", use_container_width=True, type="primary" if ss.active_tab == "Scan" else "secondary"):
            ss.active_tab = "Scan"
            ss.mood, ss.say = "listen", "Upload your breast ultrasound image."
            st.rerun()
    with t3:
        if st.button("Values", use_container_width=True, type="primary" if ss.active_tab == "Values" else "secondary"):
            ss.active_tab = "Values"
            ss.mood, ss.say = "listen", "Enter the values from your report."
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- TAB 1: CHAT (Wired to your Python RAG) ----------------
    if ss.active_tab == "Chat":
        for role, txt in ss.msgs:
            st.chat_message(role).write(txt)

        # Quick action chips
        qc1, qc2, qc3 = st.columns(3)
        user_prompt = None
        with qc1:
            if st.button("Early signs?"): user_prompt = "What are early signs of breast cancer?"
        with qc2:
            if st.button("Screening?"): user_prompt = "How often should I get screened?"
        with qc3:
            if st.button("Upload scan"): 
                ss.active_tab = "Scan"
                st.rerun()

        # Chat Input Box
        with st.form(key="chat_form", clear_on_submit=True):
            ic1, ic2 = st.columns([5, 1])
            with ic1:
                typed_msg = st.text_input("Ask Dr. Nana...", label_visibility="collapsed", placeholder="Ask Dr. Nana a health question…")
            with ic2:
                sent = st.form_submit_button("Send", type="primary", use_container_width=True)

        if sent and typed_msg:
            user_prompt = typed_msg

        if user_prompt:
            ss.msgs.append(("user", user_prompt))
            low = user_prompt.lower()
            
            # Simple responses or direct call to your Python RAG function
            if re.match(r"(hi|hello|hey|salam)", low):
                reply = "Hello! How can I help you today?"
                ss.mood, ss.say = "wave", "Hello! Nice to see you."
            elif "thank" in low:
                reply = "You're very welcome! Take care of yourself."
                ss.mood, ss.say = "good", "Happy to help!"
            else:
                with st.spinner("Dr. Nana is thinking..."):
                    try:
                        # THIS CALLS YOUR ACTUAL PYTHON RAG BACKEND!
                        reply = M.rag_answer(user_prompt)
                    except Exception as e:
                        reply = "Regular screening and knowing your body's normal are the best defenses. Tell a doctor about any new lump or change."
                ss.mood, ss.say = "talk", reply[:60] + "…"

            ss.msgs.append(("assistant", reply))
            st.rerun()

    # ---------------- TAB 2: SCAN ----------------
    elif ss.active_tab == "Scan":
        st.markdown("##### Upload Breast Ultrasound Image")
        f = st.file_uploader("Upload scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if f:
            img = Image.open(f)
            st.image(img, width=280)
            if models["resnet"]:
                with st.spinner("Analyzing image..."):
                    res = M.predict_image(models["resnet"], img)
                for k, v in res.items():
                    st.progress(float(v), text=f"{k}: {v*100:.1f}%")
            else:
                st.error("ResNet model not found.")

    # ---------------- TAB 3: VALUES ----------------
    elif ss.active_tab == "Values":
        st.markdown("##### Enter Lab Measurements")
        with st.form("val_form"):
            c = st.columns(2)
            vals = {}
            fields = [
                ("texture_mean", "Average Tissue Texture", 19.0),
                ("concave points_mean", "Average Indentations", 0.05),
                ("radius_se", "Cell Size Variability", 0.4),
                ("area_se", "Cell Area Variability", 40.0)
            ]
            for i, (k, lbl, ex) in enumerate(fields):
                vals[k] = c[i % 2].number_input(lbl, value=ex, format="%.4f")
            
            run_btn = st.form_submit_button("Analyze values", type="primary", use_container_width=True)

        if run_btn:
            if models["values"]:
                with st.spinner("Analyzing values..."):
                    pred = M.predict_values(models["values"], vals)
                for k, v in pred.items():
                    st.progress(float(v), text=f"{k}: {v*100:.1f}%")
            else:
                st.error("Values model not found.")

st.caption("AI-assisted screening support for educational use. It is not a medical diagnosis; please consult a qualified physician.")
