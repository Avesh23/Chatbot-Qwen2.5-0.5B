"""
model.py — Qwen2.5-0.5B inference
Supports:
  - Normal generation  → generate_response()
  - Streaming          → generate_response_stream()  (word by word)
  - Title generation   → generate_title()
"""

from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
from threading import Thread
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load tokenizer and model once at startup."""
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("[INFO] Loading model (this may take a minute on first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print("[INFO] Model loaded successfully.")
    return tokenizer, model


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(history: list[dict], user_input: str) -> str:
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    lines.append(f"User: {user_input}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# NORMAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    tokenizer,
    model,
    history: list[dict],
    user_input: str,
    max_new_tokens: int = 250,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> str:
    prompt    = build_prompt(history, user_input)
    inputs    = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if "User:" in response:
        response = response.split("User:")[0]
    return response.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_response_stream(
    tokenizer,
    model,
    history: list[dict],
    user_input: str,
    max_new_tokens: int = 250,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
):
    """
    Generator that yields text tokens one by one as they are produced.
    Used by the /stream SSE endpoint.
    """
    prompt   = build_prompt(history, user_input)
    inputs   = tokenizer(prompt, return_tensors="pt")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    buffer = ""
    for token_text in streamer:
        if "User:" in buffer + token_text:
            leftover  = (buffer + token_text).split("User:")[0]
            remaining = leftover[len(buffer):]
            if remaining:
                yield remaining
            break
        buffer += token_text
        yield token_text

    thread.join()


# ══════════════════════════════════════════════════════════════════════════════
# TITLE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_title(tokenizer, model, first_user_message: str) -> str:
    """
    Generate a short 4-6 word title for a session based on first user message.
    """
    prompt = (
        f"Summarize this message as a short title of 4-6 words. "
        f"Only output the title, nothing else.\n\n"
        f"Message: {first_user_message[:200]}\n"
        f"Title:"
    )
    inputs    = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    title      = tokenizer.decode(new_tokens, skip_special_tokens=True)
    title      = title.split("\n")[0].strip().strip('"').strip("'").strip(".")

    if not title or len(title) > 60:
        words = first_user_message.strip().split()
        title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")

    return title