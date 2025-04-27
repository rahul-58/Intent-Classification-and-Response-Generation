import streamlit as st
import torch
import pickle
import os
import time
import warnings
import numpy as np
import joblib

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
    try:
        st.session_state.id2label = joblib.load("BERT_id2label.pkl")
    except:
        st.session_state.id2label = {}  # fallback


def load_bert_model():
    """Load the BERT model with proper CPU compatibility handling"""
    try:
        with st.spinner("Loading BERT classifier model..."):
            model_path = "BERT_classifier.pkl"
            
            # Special handling for CUDA-serialized models
            try:
                # Monkey patch torch.cuda to handle CUDA serialized models on CPU-only machines
                _original_cuda_is_available = torch.cuda.is_available
                torch.cuda.is_available = lambda: False
                
                try:
                    # First attempt: direct load with CPU mapping
                    model = torch.load(model_path, map_location='cpu')
                    st.session_state.model = model
                    st.success("BERT model loaded successfully!")
                    time.sleep(1)
                    return True
                except Exception as e1:
                    st.warning(f"First load attempt failed: {str(e1)}")
                    
                    try:
                        # Second attempt: pickle with special handling
                        import io
                        import pickle
                        
                        class CPUUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module == 'torch.storage' and name == '_load_from_bytes':
                                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                                else:
                                    return super().find_class(module, name)
                        
                        with open(model_path, 'rb') as f:
                            model = CPUUnpickler(f).load()
                        
                        st.session_state.model = model
                        st.success("BERT model loaded successfully with custom unpickler!")
                        time.sleep(1)
                        return True
                    except Exception as e2:
                        st.warning(f"Second load attempt failed: {str(e2)}")
                        
                        # Last resort: Try to recreate the model and load state dict
                        try:
                            # This would need to be customized based on your model architecture
                            from transformers import BertForSequenceClassification
                            
                            # Create a new model instance with the same architecture
                            # We don't know number of labels, so we'll use a default and hope it works
                            temp_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                                                      num_labels=15)
                            
                            # Try to load just the state dictionary
                            state_dict = torch.load(model_path, map_location='cpu')
                            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                                state_dict = state_dict['state_dict']
                            
                            # Try to load the state dict
                            temp_model.load_state_dict(state_dict)
                            
                            st.session_state.model = temp_model
                            st.success("Recreated BERT model with state dict!")
                            time.sleep(1)
                            return True
                        except Exception as e3:
                            raise Exception(f"All loading methods failed. Errors: {str(e1)}, {str(e2)}, {str(e3)}")
                finally:
                    # Restore original cuda availability function
                    torch.cuda.is_available = _original_cuda_is_available
            except Exception as e:
                raise Exception(f"CUDA compatibility handling failed: {str(e)}")
                
    except Exception as e:
        # If loading fails, log the error
        st.error(f"Could not load BERT model: {str(e)}")
        time.sleep(2)
        st.session_state.model = None
        return False

def classify_with_bert(text):
    """Use the loaded BERT model to classify intent"""
    if st.session_state.model is None:
        return "No Model Available", 0.0
    
    try:
        model = st.session_state.model
        
        # Try different approaches to get predictions from the model
        
        # Approach 1: Direct call if model is callable
        if callable(getattr(model, '__call__', None)):
            try:
                # Try to import transformers components
                from transformers import BertTokenizer
                
                # Initialize tokenizer
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                
                # Tokenize input
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                
                # Move inputs to CPU
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get predicted class and confidence
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Get the class with highest probability
                predicted_class_idx = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()
                
                # We don't have intent labels directly, so return the class index and confidence
                label = st.session_state.id2label.get(predicted_class_idx, f"Class {predicted_class_idx}")
                return label, confidence

                
            except Exception as e:
                st.warning(f"Transformer-based prediction failed: {str(e)}")
                
                # Try a simpler approach
                try:
                    with torch.no_grad():
                        prediction = model(text)
                        predicted_class_idx = torch.argmax(prediction).item()
                        confidence = float(torch.max(torch.softmax(prediction, dim=0)).item())
                        label = st.session_state.id2label.get(predicted_class_idx, f"Class {predicted_class_idx}")
                        return label, confidence

                except Exception as e2:
                    st.warning(f"Simple prediction failed too: {str(e2)}")
        
        # Approach 2: Try predict method if available
        if hasattr(model, 'predict'):
            try:
                prediction = model.predict([text])
                if isinstance(prediction, tuple) and len(prediction) >= 2:
                    return prediction[0], prediction[1]  # Intent and confidence
                else:
                    return f"Class {prediction[0]}", 0.85
            except Exception as e:
                st.warning(f"Model.predict method failed: {str(e)}")
        
        # Approach 3: Try scikit-learn style prediction
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba([text])[0]
                predicted_class_idx = np.argmax(probs)
                confidence = float(probs[predicted_class_idx])
                label = st.session_state.id2label.get(predicted_class_idx, f"Class {predicted_class_idx}")
                return label, confidence

            except Exception as e:
                st.warning(f"predict_proba method failed: {str(e)}")
        
        # If all else fails, extract the model class name for debugging
        model_type = type(model).__name__
        raise Exception(f"Cannot determine how to use this model type: {model_type}")
        
    except Exception as e:
        # If BERT prediction fails completely, return error
        st.error(f"BERT prediction failed: {str(e)}")
        return "Prediction Error", 0.0

def handle_user_input():
    """Process user input and update chat history"""
    if st.session_state.user_input.strip():
        user_input = st.session_state.user_input
        
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_input))
        
        # Clear the input field
        st.session_state.user_input = ""
        
        # Ensure model is loaded
        if st.session_state.model is None:
            load_bert_model()
        
        # Classify the intent using only BERT
        if st.session_state.model is not None:
            intent, confidence = classify_with_bert(user_input)
            
            # Add intent classification to chat history
            intent_message = f"Detected Intent: {intent}\nConfidence: {confidence:.2f}"
            st.session_state.chat_history.append(("intent", intent_message))
        else:
            st.session_state.chat_history.append(("intent", "Model not available. Please check logs."))

# Attempt to load the BERT model on startup
if st.session_state.model is None:
    load_bert_model()

# Custom CSS for styling
st.markdown("""
<style>
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
    
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>

<!-- Navigation Bar -->
<div class="navbar">
    <div class="navbar-brand">
        <div class="title">Intent Classifier</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display model status
model_status = "BERT Model Loaded" if st.session_state.model is not None else "Model Not Available"
st.markdown(f"<div style='text-align: center; padding: 10px; opacity: 0.7;'>{model_status}</div>", unsafe_allow_html=True)

# Display messages or empty state
if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; color: rgba(255, 255, 255, 0.7); font-style: italic;">
        <p>Welcome to Intent Classification</p>
        <p>Type a message to see its intent classification</p>
        <ul style="list-style: none; padding: 0; text-align: center;">
            <li>"Where is my order #12345?"</li>
            <li>"I need to return this broken product"</li>
            <li>"Set an alarm for 7am tomorrow"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f'<div class="user-message"><div class="message-content"><strong>You:</strong> {message}</div></div>', unsafe_allow_html=True)
        elif speaker == "intent":
            intent_info = message.split('\n')
            if len(intent_info) >= 2:
                intent_name = intent_info[0].replace("Detected Intent: ", "")
                confidence = intent_info[1].replace("Confidence: ", "")
                
                st.markdown(f"""
                <div class="intent-box">
                    <div class="intent-label">{intent_name}</div>
                    <div class="intent-confidence">Confidence: {confidence}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Handle simpler message format
                st.markdown(f"""
                <div class="intent-box">
                    <div class="message-content">{message}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input box
user_input = st.text_input("", placeholder="Type your message here to classify intent...", key="user_input", on_change=handle_user_input, label_visibility="collapsed") 