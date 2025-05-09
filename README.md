# 🐬 Dolphin-7B Voice Assistant API

This is a lightweight FastAPI-based web server that wraps a local [Dolphin-2.6-Mistral-7B](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b) model (in GGUF format via `llama.cpp`) to generate polite, natural voice assistant responses.

---

## 📁 Project Structure

```
#capstone/
├── server.py           # FastAPI app exposing /generate endpoint
├── model/              # Folder with your Dolphin GGUF model
│   └── dolphin-2.6-mistral-7b.Q4_K_M.gguf
|   └── distilbert_intent_model.pkl
|   └── label_encoder.pkl
├── index.html          # Simple UI frontend to test locally
└── README.md           # This file
```

---

## 🚀 Quick Start

### 1. ✅ Install Dependencies

Make sure you have Python 3.8+ and `pip`.

```bash
pip install fastapi uvicorn llama-cpp-python
```

> 🧠 You can also create a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate  # on macOS/Linux
> venv\Scripts\activate     # on Windows
> ```

---

### 2. 📍 Place the GGUF Model

Download your model file (e.g., Dolphin-2.6-Mistral-7B) from HuggingFace or your storage and place it in:

```
/model/dolphin-2.6-mistral-7b.Q4_K_M.gguf
```

Update `MODEL_PATH` in `server.py` if needed.

---

### 3. ▶️ Start the Server

Run this command from the directory where `server.py` exists:

```bash
uvicorn server:app --reload
```

You should see:

```
Uvicorn running on http://127.0.0.1:8000
```

---

### 4. 🔍 Test the API

Go to the Swagger UI at:

```
http://127.0.0.1:8000/docs
```

Use the `/generate` endpoint by sending a `POST` request with JSON like:

```json
{
  "user_utterance": "Turn on the kitchen lights",
  "intent": "iot_hue_lighton"
}
```

You will receive a polite assistant response powered by Dolphin!

---

### 5 Run the UI

pip install -r requirements.txt
python -m streamlit run app.py

## 🧪 Example curl test

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"user_utterance": "What is the weather tomorrow?", "intent": "weather_query"}'
```

---

## 🖥 Optional: Simple Frontend

You can use the provided `index.html` file to run a minimal UI in the browser. Just open the file in Chrome or Safari.

---

## 📌 Notes

- Make sure your model fits in RAM. Quantized models like Q4_K_M are recommended.
- The server runs locally and doesn't require internet.
- Model loading may take ~30 seconds initially.

---

## 🧠 Credits

- Model: [Dolphin 2.6 Mistral 7B](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b)
- Framework: [FastAPI](https://fastapi.tiangolo.com/)
- Runtime: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

---
