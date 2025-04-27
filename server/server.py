# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from functools import lru_cache
import os
import multiprocessing
import logging
import time

# === Setup logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# === Setup Llama model ===
MODEL_PATH = "model/dolphin-2.6-mistral-7b.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

@lru_cache(maxsize=1)
def get_llm():
    n_threads = multiprocessing.cpu_count()  # auto-detect CPU cores
    logging.info(f"Initializing Llama with {n_threads} threads...")
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=256,         # 256 context for faster inference
        n_threads=n_threads,
        use_mmap=True,     # memory map model into RAM for speed
        use_mlock=True     # try to prevent swapping (needs sudo sometimes)
    )

llm = get_llm()

# === Request and Response Schemas ===
class QueryRequest(BaseModel):
    user_utterance: str
    intent: str

class QueryResponse(BaseModel):
    response: str

# === Generate Response Function ===
def generate_response(user_utterance, intent):
    if intent.lower() == "other":
        prompt = (
            f"[INST] <<SYS>> You are a polite voice assistant. If the user request cannot be handled, respond with a short, polite rejection sentence only. <<SYS>>\n"
            f"User: {user_utterance}\n"
            f"Intent: {intent}\n"
            f"Assistant:"
        )
    else:
        prompt = (
            f"[INST] <<SYS>> You are a polite voice assistant. Confirm the action politely in one sentence only. <<SYS>>\n"
            f"User: {user_utterance}\n"
            f"Intent: {intent}\n"
            f"Assistant:"
        )

    start_time = time.time()

    # generate output
    output = llm(
        prompt,
        max_tokens=60,
        temperature=0.5,
        top_p=0.85,
        stop=["</s>", "[/INST]"]
    )

    elapsed_time = time.time() - start_time
    raw_response = output["choices"][0]["text"].strip()
    first_sentence = raw_response.split(".")[0].strip()
    final_response = first_sentence + "." if first_sentence else "I'm sorry, I couldn't process that request. Please try again."

    # Logging
    logging.info(f"User utterance: {user_utterance}")
    logging.info(f"Intent: {intent}")
    logging.info(f"Assistant response: {final_response}")
    logging.info(f"Inference took {elapsed_time:.2f} seconds.")

    return final_response

# === FastAPI Endpoint ===
@app.post("/generate", response_model=QueryResponse)
def generate_text(request: QueryRequest):
    response = generate_response(request.user_utterance, request.intent)
    return QueryResponse(response=response)

@app.get("/")
def read_root():
    return {"message": "Welcome! Use /docs endpoint for more info."}
