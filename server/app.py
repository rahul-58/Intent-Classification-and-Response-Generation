import streamlit as st
import torch
import pickle
import time
import numpy as np
import joblib
import io
import requests

FASTAPI_SERVER_URL = "http://127.0.0.1:8000/generate"

# Set page configuration
st.set_page_config(
    page_title="Intent Classification",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "id2label" not in st.session_state:
    st.session_state.id2label = {}  # We'll fill after loading model

if "label_encoder" not in st.session_state:
    try:
        with open('model/label_encoder.pkl', 'rb') as f:
            st.session_state.label_encoder = pickle.load(f)
    except Exception as e:
        st.session_state.label_encoder = None
        st.error(f"Label encoder loading failed: {e}")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_model_and_id2label():
    """Load the GPU-trained model correctly onto CPU"""
    try:
        with st.spinner("Loading model..."):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress to improve user experience
            for i in range(101):
                # Update progress bar
                progress_bar.progress(i)
                
                # Update status text based on progress
                if i < 30:
                    status_text.markdown("⚙️ Initializing model components...")
                elif i < 60:
                    status_text.markdown("📦 Loading model weights...")
                elif i < 90:
                    status_text.markdown("🔄 Configuring model parameters...")
                else:
                    status_text.markdown("✅ Finalizing model setup...")
                
                # Only sleep for a short time to simulate work
                time.sleep(0.02)

            model_path = "model/distilbert_intent_model.pkl"
            
            with open(model_path, 'rb') as f:
                model = CPU_Unpickler(f).load()

            st.session_state.model = model

            # Try extracting id2label if available
            if hasattr(model, "config") and hasattr(model.config, "id2label"):
                st.session_state.id2label = model.config.id2label
            else:
                num_labels = getattr(model, "num_labels", 10)
                st.session_state.id2label = {i: f"Class {i}" for i in range(num_labels)}
            
            # print(model.config.label, model)

            st.success("✅ Model loaded successfully!")
            return True

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.session_state.model = None
        return False


def classify_intent(text):
    """Classify user input and return label + confidence"""
    if st.session_state.model is None:
        return "No Model Available", 0.0
    
    model = st.session_state.model
    id2label = st.session_state.id2label

    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()

        if st.session_state.get("label_encoder", None) is not None:
            try:
                intent_name = st.session_state.label_encoder.inverse_transform([predicted_idx])[0]
            except Exception as e:
                intent_name = f"Class {predicted_idx}"
        else:
            intent_name = id2label.get(predicted_idx, f"Class {predicted_idx}")
        
        print(intent_name)

        return intent_name, confidence

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Prediction Error", 0.0

def handle_user_input():
    """Handle user input and append to chat"""
    if st.session_state.user_input.strip():
        user_input = st.session_state.user_input
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.user_input = ""

        if st.session_state.model is None:
            load_model_and_id2label()

        if st.session_state.model is not None:
            label, confidence = classify_intent(user_input)
            intent_message = f"Detected Intent: {label}\nConfidence: {confidence:.2f}"
            st.session_state.chat_history.append(("intent", intent_message))

            try:
                payload = {
                    "user_utterance": user_input,
                    "intent": label
                }
                response = requests.post(FASTAPI_SERVER_URL, json=payload)
                if response.status_code == 200:
                    server_response = response.json()["response"]
                    # 3. Append server response
                    st.session_state.chat_history.append(("assistant", server_response))
                else:
                    st.session_state.chat_history.append(("assistant", "Failed to get server response."))
            except Exception as e:
                st.session_state.chat_history.append(("assistant", f"Error contacting server: {e}"))
        else:
            st.session_state.chat_history.append(("intent", "Model not available."))

# Load model at the start
if st.session_state.model is None:
    load_model_and_id2label()

# UI Styling (your previous CSS, keep it same)
st.markdown("""
<style>
/* Your existing CSS unchanged */

        /* Main container styling */
.main {
    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    color: #FFFFFF;
    font-family: 'Amazon Ember', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Chat message styling */
.user-message {
    background-color: rgba(255, 153, 0, 0.1);
    border-radius: 12px;
    padding: 16px;
    margin: 12px auto;
    max-width: 80%;
    border-left: 4px solid #FF9900;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    word-wrap: break-word;
}

/* Intent classification result */
.intent-box {
    background-color: rgba(16, 163, 127, 0.1);
    border-radius: 12px;
    padding: 16px;
    margin: 12px auto;
    max-width: 80%;
    border-left: 4px solid #10a37f;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    word-wrap: break-word;
}

.intent-label {
    font-weight: 600;
    color: #10a37f;
    margin-bottom: 8px;
}

.intent-confidence {
    font-size: 14px;
    opacity: 0.8;
}

/* Fixed input box */
.stTextInput {
    position: fixed !important;
    bottom: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 80% !important;
    max-width: 800px !important;
    z-index: 1000 !important;
    background-color: rgba(0, 0, 0, 0.3) !important;
    border-radius: 20px !important;
    padding: 10px 20px !important;
}

/* Navbar styling */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    background-color: rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(15px);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
    height: 70px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.title {
    font-size: 22px;
    font-weight: 600;
    background: linear-gradient(90deg, #FF9900, #FFC300);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Chat container */
.chat-container {
    margin-top: 90px;
    margin-bottom: 120px;
    padding: 0 20px;
    overflow-y: auto;
    height: calc(100vh - 210px);
}
            
.main {
    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    color: #FFFFFF;
}
.user-message {
    background-color: rgba(255, 153, 0, 0.1);
    border-left: 4px solid #FF9900;
    border-radius: 10px;
    padding: 12px;
    margin: 10px;
}
.intent-box {
    background-color: rgba(16, 163, 127, 0.1);
    border-left: 4px solid #10a37f;
    border-radius: 10px;
    padding: 12px;
    margin: 10px;
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Chat Display
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align: center; padding: 40px;">
        <h3>Welcome to Intent Classifier</h3>
        <p>Type your query to classify its intent 🚀</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
        elif speaker == "intent":
            lines = message.split('\n')
            if len(lines) >= 2:
                st.markdown(f"""
                <div class="intent-box">
                    <div><strong>{lines[0]}</strong></div>
                    <div style="font-size: small; opacity: 0.7;">{lines[1]}</div>
                </div>
                """, unsafe_allow_html=True)
        elif speaker == "assistant":
            st.markdown(f'<div class="intent-box"><div class="message-content"><strong>Assistant:</strong> {message}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input Box
user_input = st.text_input("", placeholder="Type your message...", key="user_input", on_change=handle_user_input, label_visibility="collapsed")


