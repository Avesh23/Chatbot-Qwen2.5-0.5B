"""
Qwen2.5-0.5B · Story Generator — Streamlit Frontend
----------------------------------------------------
ChatGPT-style sidebar with full session history from MongoDB.

Run order:
  Terminal 1 → uvicorn api:app --host 0.0.0.0 --port 8000 --reload
  Terminal 2 → streamlit run app.py
"""

import streamlit as st
import requests
import uuid
import re
import html as html_lib
import json
from datetime import datetime
from collections import Counter

API_URL = "http://localhost:8000"

def clean_content(text: str) -> str:
    import re as _re
    text = _re.sub(r'<[^>]+>', '', text)
    text = html_lib.escape(text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATORS
# ══════════════════════════════════════════════════════════════════════════════

MIN_CHARS  = 3
MAX_CHARS  = 100000  # effectively no limit — chunking handles any size automatically
CHUNK_SIZE = 800     # max chars per chunk sent to model (~200 tokens each)

ALLOWED_SHORT = {
    "hi", "hey", "ok", "yo", "go", "no", "yes", "bye",
    "wow", "lol", "hmm", "hm", "ah", "oh", "aw", "ugh",
    "help", "sure", "cool", "nice", "good", "bad", "sad",
    "hi!", "hey!", "ok!", "hello", "thanks", "thx", "ty",
    "sup", "yep", "nah", "huh", "yay", "aww", "omg",
}

BAD_WORDS = [
    "fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap",
    "dick", "pussy", "cock", "ass", "piss", "slut", "whore", "nigger",
    "faggot", "retard", "cunt", "motherfucker", "fucker",
]

def validate_input(text: str) -> tuple[bool, str, str]:
    text = text.strip()
    if not text:
        return False, "Empty Message", "Please type something before sending!"
    is_allowed_short = text.lower().rstrip("!?.") in ALLOWED_SHORT or text.lower() in ALLOWED_SHORT
    if len(text) < MIN_CHARS and not is_allowed_short:
        return False, "Too Short", f"Your message is too short! Please write at least {MIN_CHARS} characters.\n\nYou typed: {len(text)} character(s).\n\nTip: Short greetings like 'hi' or 'hey' are fine!"
    # No hard upper limit — smart chunking splits any length automatically
    # We only warn in the counter UI, we never block long inputs
    text_lower = text.lower()
    found_bad  = []
    for word in BAD_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, text_lower):
            found_bad.append(word)
    if found_bad:
        return False, "Inappropriate Content", "Your message contains inappropriate language.\n\nPlease keep it clean and respectful! 🙏"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii / len(text)
    if non_ascii_ratio > 0.4:
        return False, "English Only", "This app only supports English input.\n\nPlease write your story prompt in English! 🇬🇧"
    return True, "", ""


# ══════════════════════════════════════════════════════════════════════════════
# SMART CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Splits long text into chunks of max chunk_size characters.
    Always cuts at sentence endings (. ! ?) so we never split mid-sentence.

    Example:
      "The knight walked. He saw a dragon. It roared." with chunk_size=30
      → ["The knight walked. He saw a dragon.", "It roared."]
    """
    if len(text) <= chunk_size:
        return [text]

    chunks    = []
    current   = ""
    sentences = re.split(r'(?<=[.!?])\s+', text)   # split at sentence endings

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks


def process_chunks(chunks: list[str], history: list, settings: dict, session_id: str) -> str:
    """
    Sends each chunk to FastAPI /generate one by one.
    Each chunk's response is added to history so the next chunk
    has full context and the story stays coherent.

    Returns all responses joined together as one string.
    """
    combined_response = ""
    working_history   = history.copy()

    for i, chunk in enumerate(chunks):
        # First chunk: send as-is
        # Later chunks: tell model it's a continuation
        if i == 0:
            prompt = chunk
        else:
            prompt = f"Continue based on this next part: {chunk}"

        result = call_generate(
            user_input=prompt,
            history=working_history,
            settings=settings,
            session_id=session_id,
        )

        if result and "error" not in result:
            chunk_response = result["response"]
            combined_response += chunk_response + "\n\n"
            # Add this exchange to working history for next chunk's context
            working_history.append({"role": "user",      "content": prompt})
            working_history.append({"role": "assistant", "content": chunk_response})
        else:
            combined_response += f"[Part {i+1} could not be processed]\n\n"

    return combined_response.strip()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qwen2.5 · Story AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Lora:ital,wght@0,400;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #0f1117 !important;
    color: #e1e4e8;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: #0a0c10 !important;
    border-right: 1px solid #1a1d2e !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 16px 14px;
    border-bottom: 1px solid #1a1d2e;
}
.sb-logo-icon {
    width: 30px; height: 30px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    border-radius: 8px; color: white;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}
.sb-logo-text  { font-size: 0.82rem; font-weight: 600; color: #e1e4e8; }
.sb-logo-sub   { font-size: 0.65rem; color: #6b7280; }

.new-chat-wrap { padding: 10px 12px 6px; }

.sb-section {
    padding: 12px 16px 4px;
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.12em; color: #374151;
    text-transform: uppercase;
}

.sess-item {
    display: flex; align-items: center; justify-content: space-between;
    margin: 1px 8px;
    border-radius: 8px; padding: 8px 10px;
    cursor: pointer; transition: background 0.15s;
}
.sess-item:hover   { background: #13151f; }
.sess-item.active  { background: #1a1d2e; border: 1px solid #2d3148; }
.sess-title {
    font-size: 0.78rem; color: #c9cdd4;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 160px;
}
.sess-meta  { font-size: 0.62rem; color: #4b5563; margin-top: 2px; }
.sess-dot   {
    width: 6px; height: 6px; border-radius: 50%;
    background: #6366f1; flex-shrink: 0; margin-right: 6px;
}

.api-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.65rem; padding: 3px 8px;
    border-radius: 20px; font-weight: 500;
}
.api-on  { background:#0d1f12; border:1px solid #14532d; color:#4ade80; }
.api-off { background:#1f0d0d; border:1px solid #7f1d1d; color:#f87171; }
.dot-pulse { width:6px;height:6px;border-radius:50%;background:#4ade80;animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ══ TOP BAR ══ */
.topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding: 16px 0 12px;
    border-bottom: 1px solid #1a1d2e;
    flex-shrink: 0;
}
.topbar-left { display:flex; align-items:center; gap:8px; }
.topbar-icon {
    width:32px; height:32px;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    border-radius:9px; font-size:15px; color:white;
    display:flex; align-items:center; justify-content:center;
}
.topbar-title { font-size:0.92rem; font-weight:600; color:#f0f1f3; }
.chip {
    font-size:0.65rem; padding:2px 8px; border-radius:20px; font-weight:500;
}
.chip-model { background:#6366f1; color:#fff; }
.chip-api-on  { background:#0d1f12; border:1px solid #14532d; color:#4ade80; }
.chip-api-off { background:#1f0d0d; border:1px solid #7f1d1d; color:#f87171; }
.topbar-right { display:flex; align-items:center; gap:6px; font-size:0.7rem; color:#6b7280; }
.live-dot { width:6px;height:6px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite; }

/* ══ CHUNK INFO BANNER ══ */
.chunk-banner {
    background:#0d1220; border:1px solid #2d3148;
    border-radius:8px; padding:8px 12px;
    font-size:0.72rem; color:#818cf8;
    margin-bottom:6px; text-align:center;
}

/* ══ CONTINUE BUTTON ══ */
.continue-wrap .stButton > button {
    background: #13151f !important;
    border: 1px solid #6366f1 !important;
    color: #a5b4fc !important;
    height: 32px !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
    padding: 0 14px !important;
    width: auto !important;
}
.continue-wrap .stButton > button:hover {
    background: #1a1d2e !important;
    border-color: #8b5cf6 !important;
    transform: none !important;
}

/* ══ WELCOME SCREEN ══ */
.welcome-wrap {
    display:flex; flex-direction:column;
    align-items:center; padding:50px 20px; text-align:center;
}
.welcome-icon {
    width:60px; height:60px;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    border-radius:18px; font-size:26px; color:white;
    display:flex; align-items:center; justify-content:center;
    margin-bottom:18px;
    box-shadow:0 0 40px rgba(99,102,241,0.2);
}
.welcome-title { font-size:1.4rem; font-weight:600; color:#f0f1f3; margin-bottom:6px; }
.welcome-sub   { font-size:0.82rem; color:#6b7280; max-width:380px; line-height:1.65; margin-bottom:28px; }
.sug-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; max-width:500px; width:100%; }
.sug-card {
    background:#13151f; border:1px solid #1a1d2e;
    border-radius:11px; padding:12px 14px; text-align:left;
    transition: border-color 0.2s;
}
.sug-card:hover { border-color: #6366f1; }
.sug-emoji { font-size:1rem; margin-bottom:5px; }
.sug-text  { font-size:0.75rem; color:#9ca3af; line-height:1.45; }

/* ══ CHAT BUBBLES ══ */
.msg-row { display:flex; gap:10px; margin-bottom:20px; animation:fadeUp 0.25s ease; }
.msg-row.user-row { flex-direction:row-reverse; }
@keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

.avatar {
    width:32px; height:32px; border-radius:9px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    font-size:13px; font-weight:600;
}
.av-user  { background:#1e2130; color:#818cf8; border:1px solid #2d3148; }
.av-model { background:linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; }

.bub-wrap { display:flex; flex-direction:column; max-width:75%; }
.user-row .bub-wrap { align-items:flex-end; }

.bub-name { font-size:0.65rem; font-weight:600; color:#6b7280; margin-bottom:4px; }
.bub {
    padding:11px 15px; border-radius:14px;
    font-size:0.87rem; line-height:1.75;
    word-wrap:break-word; white-space:pre-wrap;
}
.bub-user  { background:#6366f1; color:#fff; border-bottom-right-radius:3px; }
.bub-model {
    background:#13151f; border:1px solid #1a1d2e;
    color:#d1d5db; border-bottom-left-radius:3px;
    font-family:'Lora',serif; font-size:0.88rem;
}
.bub-meta { font-size:0.62rem; color:#374151; margin-top:4px; padding:0 3px; }

/* Typing dots */
.gen-row { display:flex; gap:10px; margin-bottom:18px; }
.typing-bub {
    background:#13151f; border:1px solid #1a1d2e;
    border-radius:14px; border-bottom-left-radius:3px;
    padding:13px 16px; display:flex; align-items:center; gap:4px;
}
.dot { width:6px;height:6px;background:#6366f1;border-radius:50%;animation:bounce 1.2s infinite; }
.dot:nth-child(2){animation-delay:.2s} .dot:nth-child(3){animation-delay:.4s}
@keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-5px)} }

/* ══ ERROR ══ */
.err-banner {
    background:#1f0d0d; border:1px solid #7f1d1d;
    border-radius:10px; padding:11px 14px;
    font-size:0.8rem; color:#f87171; margin-bottom:14px;
}

/* ══ INPUT BAR ══ */
.input-hint {
    font-size:0.65rem; color:#374151; text-align:center;
    padding:8px 0 5px; border-top:1px solid #1a1d2e; margin-top:6px;
}
[data-testid="stTextArea"] label { display:none !important; }
[data-testid="stTextArea"] textarea {
    background:#13151f !important;
    border:1.5px solid #1a1d2e !important;
    border-radius:13px !important;
    color:#e1e4e8 !important;
    font-size:0.87rem !important;
    font-family:'Inter',sans-serif !important;
    line-height:1.6 !important;
    padding:11px 13px !important;
    resize:none !important;
    transition:border-color 0.2s !important;
    box-shadow:none !important;
    outline:none !important;
}
[data-testid="stTextArea"] textarea:focus { border-color:#6366f1 !important; }
[data-testid="stTextArea"] textarea::placeholder { color:#2d3148 !important; }
[data-testid="stTextArea"] > div { border:none !important; background:transparent !important; box-shadow:none !important; }

/* Send button */
.stButton > button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:white !important; border:none !important;
    border-radius:11px !important;
    height:50px !important; width:100% !important;
    font-size:1.05rem !important;
    transition:opacity 0.2s, transform 0.1s !important;
    min-height:0 !important;
}
.stButton > button:hover  { opacity:0.85 !important; transform:scale(1.03) !important; }
.stButton > button:active { transform:scale(0.97) !important; }

/* New chat / clear buttons */
.btn-new .stButton > button {
    background:#13151f !important;
    border:1px solid #2d3148 !important;
    color:#a5b4fc !important;
    height:36px !important;
    font-size:0.78rem !important;
    border-radius:8px !important;
}
.btn-new .stButton > button:hover {
    background:#1a1d2e !important;
    border-color:#6366f1 !important;
    transform:none !important;
}

/* Session load buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #c9cdd4 !important;
    height: auto !important;
    min-height: 52px !important;
    padding: 8px 10px !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    border-radius: 8px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    line-height: 1.4 !important;
    transition: background 0.15s, border-color 0.15s !important;
    transform: none !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #13151f !important;
    border-color: #2d3148 !important;
    opacity: 1 !important;
    transform: none !important;
}
[data-testid="stSidebar"] .btn-new .stButton > button {
    background: #13151f !important;
    border: 1px solid #2d3148 !important;
    color: #a5b4fc !important;
    height: 36px !important;
    min-height: 36px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    text-align: center !important;
    justify-content: center !important;
}
.btn-del .stButton > button {
    background:transparent !important;
    border:1px solid #2a1515 !important;
    color:#ef4444 !important;
    height:32px !important;
    font-size:0.72rem !important;
    border-radius:7px !important;
    opacity:0.7 !important;
}
.btn-del .stButton > button:hover {
    background:#1a0f0f !important;
    border-color:#ef4444 !important;
    opacity:1 !important;
    transform:none !important;
}

[data-testid="stSpinner"] { display:none !important; }

/* ══ VALIDATION POPUP ══ */
.popup-overlay {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    z-index: 9998;
    display: flex; align-items: center; justify-content: center;
    animation: fadeInOverlay 0.2s ease;
}
@keyframes fadeInOverlay { from{opacity:0} to{opacity:1} }

.popup-box {
    background: #13151f;
    border: 1px solid #2d3148;
    border-radius: 18px;
    padding: 28px 32px;
    max-width: 380px; width: 90%;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    animation: popIn 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    z-index: 9999;
}
@keyframes popIn {
    from{opacity:0; transform:scale(0.85) translateY(10px)}
    to  {opacity:1; transform:scale(1)    translateY(0)}
}
.popup-icon  { font-size: 2.5rem; margin-bottom: 12px; }
.popup-title {
    font-size: 1.05rem; font-weight: 700;
    color: #f0f1f3; margin-bottom: 10px;
}
.popup-msg {
    font-size: 0.82rem; color: #9ca3af;
    line-height: 1.65; margin-bottom: 22px;
    white-space: pre-wrap;
}
.popup-btn {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    padding: 9px 28px !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.popup-btn:hover { opacity: 0.85 !important; }

/* ✕ top-right close icon */
.popup-x {
    position: absolute; top: 12px; right: 16px;
    font-size: 1.1rem; color: #4b5563;
    cursor: pointer; line-height: 1;
    transition: color 0.15s;
    user-select: none;
}
.popup-x:hover { color: #e1e4e8; }

/* popup-box needs position:relative for the ✕ to anchor to */
.popup-box { position: relative; }

/* "OK, got it" button inside popup */
.popup-ok-btn {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white; border: none;
    border-radius: 10px;
    padding: 9px 32px;
    font-size: 0.85rem; font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
    width: 100%;
}
.popup-ok-btn:hover { opacity: 0.85; }

/* Char counter */
.char-counter {
    font-size: 0.65rem; text-align: right;
    padding: 3px 4px 0;
    transition: color 0.2s;
}
.char-ok   { color: #374151; }
.char-warn { color: #f59e0b; }
.char-over { color: #ef4444; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# API HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        return r.status_code == 200
    except:
        return False


def fetch_sessions() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/sessions", timeout=5)
        if r.status_code == 200:
            return r.json().get("sessions", [])
    except:
        pass
    return []


def fetch_session_messages(session_id: str) -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/history/{session_id}", timeout=5)
        if r.status_code == 200:
            return r.json().get("messages", [])
    except:
        pass
    return []


def call_generate(user_input: str, history: list, settings: dict, session_id: str) -> dict | None:
    payload = {
        "user_input":         user_input,
        "history":            history,
        "session_id":         session_id,
        "max_new_tokens":     settings["max_tokens"],
        "temperature":        settings["temperature"],
        "top_p":              settings["top_p"],
        "repetition_penalty": settings["rep_penalty"],
    }
    try:
        r = requests.post(f"{API_URL}/generate", json=payload, timeout=120)
        return r.json() if r.status_code == 200 else {"error": r.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to FastAPI. Is it running on port 8000?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Model is taking too long on CPU."}
    except Exception as e:
        return {"error": str(e)}


def count_tokens(text: str) -> dict:
    """
    Call /count-tokens endpoint to get EXACT token count from the real tokenizer.
    Returns dict with token_count, char_count, word_count.
    Falls back to estimate if API is unavailable.
    """
    if not text or not text.strip():
        return {"token_count": 0, "char_count": 0, "word_count": 0}
    try:
        r = requests.post(f"{API_URL}/count-tokens", json={"text": text}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    # Fallback estimate if API call fails
    return {
        "token_count": max(1, len(text) // 4),
        "char_count":  len(text),
        "word_count":  len(text.split()),
    }


def delete_session(session_id: str) -> bool:
    try:
        r = requests.delete(f"{API_URL}/history/{session_id}", timeout=5)
        return r.status_code == 200
    except:
        return False


def call_stream(user_input: str, history: list, settings: dict, session_id: str):
    """Call /stream SSE endpoint and yield tokens one by one."""
    payload = {
        "user_input":         user_input,
        "history":            history,
        "session_id":         session_id,
        "max_new_tokens":     settings["max_tokens"],
        "temperature":        settings["temperature"],
        "top_p":              settings["top_p"],
        "repetition_penalty": settings["rep_penalty"],
    }
    try:
        with requests.post(f"{API_URL}/stream", json=payload, stream=True, timeout=180) as r:
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        yield data
    except Exception as e:
        yield {"token": f"[Error: {e}]", "done": True}


def save_rating(msg_id: str, session_id: str, rating: int) -> bool:
    try:
        r = requests.post(f"{API_URL}/rate", json={
            "msg_id": msg_id, "session_id": session_id, "rating": rating
        }, timeout=5)
        return r.status_code == 200
    except:
        return False


def fetch_session_ratings(session_id: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/ratings/{session_id}", timeout=5)
        if r.status_code == 200:
            return {x["msg_id"]: x["rating"] for x in r.json().get("ratings", [])}
    except:
        pass
    return {}


def fetch_stats() -> dict:
    try:
        r = requests.get(f"{API_URL}/stats", timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}


def fetch_ai_title(session_id: str) -> str | None:
    try:
        r = requests.get(f"{API_URL}/title/{session_id}", timeout=60)
        if r.status_code == 200:
            return r.json().get("title")
    except:
        pass
    return None


def format_session_title(messages: list, session_id: str) -> str:
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            return content[:40] + "…" if len(content) > 40 else content
    return session_id


def format_time(ts_str: str) -> str:
    try:
        ts   = datetime.fromisoformat(ts_str)
        now  = datetime.now()
        diff = (now - ts).days
        if diff == 0:   return "Today"
        if diff == 1:   return "Yesterday"
        if diff < 7:    return f"{diff} days ago"
        return ts.strftime("%b %d")
    except:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "history"            not in st.session_state: st.session_state.history            = []
if "session_id"         not in st.session_state: st.session_state.session_id         = f"sess-{str(uuid.uuid4())[:8]}"
if "generating"         not in st.session_state: st.session_state.generating         = False
if "input_key"          not in st.session_state: st.session_state.input_key          = 0
if "pending"            not in st.session_state: st.session_state.pending            = None
if "last_meta"          not in st.session_state: st.session_state.last_meta          = None
if "sessions"           not in st.session_state: st.session_state.sessions           = []
if "sess_refresh"       not in st.session_state: st.session_state.sess_refresh       = 0
if "popup"              not in st.session_state: st.session_state.popup              = None
if "page"               not in st.session_state: st.session_state.page               = "chat"
if "ratings"            not in st.session_state: st.session_state.ratings            = {}
if "stream_text"        not in st.session_state: st.session_state.stream_text        = ""
if "ai_title_req"       not in st.session_state: st.session_state.ai_title_req       = None
if "is_chunking"        not in st.session_state: st.session_state.is_chunking        = False
if "continue_requested" not in st.session_state: st.session_state.continue_requested = False
if "input_token_count"  not in st.session_state: st.session_state.input_token_count  = 0


# ══════════════════════════════════════════════════════════════════════════════
# LOAD SESSIONS FROM MONGODB
# ══════════════════════════════════════════════════════════════════════════════
api_ok = check_api()
if api_ok:
    st.session_state.sessions = fetch_sessions()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(f"""
    <div class="sb-logo">
        <div class="sb-logo-icon">✦</div>
        <div>
            <div class="sb-logo-text">Avesh ChatBot Qwen2.5-0.5B</div>
            <div class="sb-logo-sub">Story Generator</div>
        </div>
        <div style="margin-left:auto">
            <span class="api-pill {'api-on' if api_ok else 'api-off'}">
                {'<span class="dot-pulse"></span> Live' if api_ok else '✗ Offline'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Nav buttons ────────────────────────────────────────────────────────
    col_chat, col_dash = st.columns(2)
    with col_chat:
        if st.button("💬 Chat", use_container_width=True, key="nav_chat"):
            st.session_state.page = "chat"
            st.rerun()
    with col_dash:
        if st.button("📊 Stats", use_container_width=True, key="nav_dash"):
            st.session_state.page = "dashboard"
            st.rerun()

    # RAG nav button
    if st.button("📚 Documents", use_container_width=True, key="nav_rag"):
        st.session_state.page = "rag"
        st.rerun()

    st.markdown('<div class="new-chat-wrap">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="btn-new">', unsafe_allow_html=True)
        if st.button("✦  New conversation", use_container_width=True, key="new_chat"):
            st.session_state.history            = []
            st.session_state.session_id         = f"sess-{str(uuid.uuid4())[:8]}"
            st.session_state.generating         = False
            st.session_state.pending            = None
            st.session_state.last_meta          = None
            st.session_state.continue_requested = False
            st.session_state.input_key         += 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sessions = st.session_state.sessions

    if sessions:
        groups = {"Today": [], "Yesterday": [], "Earlier": []}
        for s in sessions:
            label = format_time(s.get("last_updated", ""))
            if label == "Today":       groups["Today"].append(s)
            elif label == "Yesterday": groups["Yesterday"].append(s)
            else:                      groups["Earlier"].append(s)

        for group_name, group_sessions in groups.items():
            if not group_sessions:
                continue
            st.markdown(f'<div class="sb-section">{group_name}</div>', unsafe_allow_html=True)

            for s in group_sessions:
                sid        = s["session_id"]
                msg_count  = s["message_count"]
                time_label = format_time(s.get("last_updated", ""))
                is_active  = sid == st.session_state.session_id

                title_key = f"title_{sid}"
                if title_key not in st.session_state:
                    msgs = fetch_session_messages(sid)
                    st.session_state[title_key] = format_session_title(msgs, sid)
                title = st.session_state[title_key]

                col_sess, col_del = st.columns([5, 1])

                with col_sess:
                    if st.button(
                        title,
                        key=f"load_{sid}",
                        use_container_width=True,
                        help=f"Load: {title}",
                    ):
                        msgs = fetch_session_messages(sid)
                        st.session_state.history            = [{"role": m["role"], "content": m["content"]} for m in msgs]
                        st.session_state.session_id         = sid
                        st.session_state.generating         = False
                        st.session_state.pending            = None
                        st.session_state.continue_requested = False
                        st.session_state.input_key         += 1
                        st.session_state.page               = "chat"
                        st.session_state.ratings            = fetch_session_ratings(sid)
                        st.rerun()

                with col_del:
                    st.markdown('<div class="btn-del">', unsafe_allow_html=True)
                    if st.button("🗑", key=f"del_{sid}", help="Delete session"):
                        delete_session(sid)
                        if title_key in st.session_state:
                            del st.session_state[title_key]
                        if sid == st.session_state.session_id:
                            st.session_state.history    = []
                            st.session_state.session_id = f"sess-{str(uuid.uuid4())[:8]}"
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:16px 16px 8px; font-size:0.75rem; color:#374151; text-align:center; line-height:1.6;">
            No conversations yet.<br>Start chatting to see history here.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⚙  Settings", expanded=False):
        max_tokens  = st.slider("Max tokens",          50,  500, 200, step=25)
        temperature = st.slider("Temperature",        0.1,  1.5, 0.8, step=0.05)
        top_p       = st.slider("Top-p",              0.1,  1.0, 0.9, step=0.05)
        rep_penalty = st.slider("Repetition penalty", 1.0,  2.0, 1.2, step=0.05)
        st.markdown("---")
        rag_mode    = st.toggle(
            "🔬 RAG Mode — Answer from your documents",
            value=False,
            help="When ON: searches your uploaded documents and gives context to Qwen before answering."
        )
        if rag_mode:
            rag_top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 3, step=1,
                                  help="How many document pieces to use as context. More = richer context but slower.")
            st.info("📚 Upload documents in the **Documents** page first.", icon="ℹ️")
        else:
            rag_top_k = 3

try:
    max_tokens
except NameError:
    max_tokens  = 200
    temperature = 0.8
    top_p       = 0.9
    rep_penalty = 1.2
    rag_mode    = False
    rag_top_k   = 3


# ══════════════════════════════════════════════════════════════════════════════
# POPUP — Native Streamlit dialog (actually works, no JS tricks needed)
# ══════════════════════════════════════════════════════════════════════════════

@st.dialog(" ")   # empty title — we show our own styled title inside
def show_validation_popup(icon, title, msg):
    """
    Proper Streamlit modal dialog.
    st.dialog() is a real native popup — has its own X button built in.
    Clicking X or the OK button both close it cleanly.
    No JS, no hidden buttons, no refresh needed.
    """
    st.markdown(f"""
    <div style="text-align:center; padding: 8px 0 4px;">
        <div style="font-size:2.2rem; margin-bottom:10px">{icon}</div>
        <div style="font-size:1.05rem; font-weight:700; color:#f0f1f3;
                    margin-bottom:10px">{title}</div>
        <div style="font-size:0.85rem; color:#9ca3af; line-height:1.7;
                    white-space:pre-wrap; margin-bottom:20px">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Real Streamlit button — clicking this actually works
    if st.button("OK, got it", use_container_width=True, key="popup_ok"):
        st.session_state.popup = None
        st.rerun()

# Show the dialog if popup is set
if st.session_state.popup:
    p = st.session_state.popup
    show_validation_popup(p["icon"], p["title"], p["msg"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Top bar
# ══════════════════════════════════════════════════════════════════════════════
current_title = "New Conversation"
if st.session_state.history:
    for m in st.session_state.history:
        if m["role"] == "user":
            t = m["content"]
            current_title = (t[:35] + "…") if len(t) > 35 else t
            break

st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-icon">✦</div>
        <div class="topbar-title">{current_title}</div>
        <span class="chip chip-model">Qwen2.5-0.5B</span>
        <span class="chip {'chip-api-on' if api_ok else 'chip-api-off'}">
            {'✓ FastAPI' if api_ok else '✗ API Offline'}
        </span>
    </div>
    <div class="topbar-right">
        <div class="live-dot"></div>
        Session: {st.session_state.session_id}
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — API offline warning
# ══════════════════════════════════════════════════════════════════════════════
if not api_ok:
    st.markdown("""
    <div class="err-banner">
        ⚠️ <strong>FastAPI server is not running.</strong>
        Open a terminal and run:
        <code style="background:#2a0f0f;padding:2px 6px;border-radius:4px;">
            uvicorn api:app --host 0.0.0.0 --port 8000 --reload
        </code>
        then refresh this page.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Chat messages
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.history and not st.session_state.generating:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">✦</div>
        <div class="welcome-title">Qwen2.5-0.5B Story Generator</div>
        <div class="welcome-sub">Start a story, continue one, or paste a long text — smart chunking handles it automatically.</div>
        <div class="sug-grid">
            <div class="sug-card">
                <div class="sug-emoji">🏰</div>
                <div class="sug-text">Write a story about a knight who discovers a hidden kingdom beneath the ocean.</div>
            </div>
            <div class="sug-card">
                <div class="sug-emoji">🚀</div>
                <div class="sug-text">Write a story about the first human to discover life on Mars.</div>
            </div>
            <div class="sug-card">
                <div class="sug-emoji">🌊</div>
                <div class="sug-text">She had been lost at sea for thirty days when the island appeared.</div>
            </div>
            <div class="sug-card">
                <div class="sug-emoji">🤖</div>
                <div class="sug-text">The robot sat alone in the museum, long after the last human had gone.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    for turn in st.session_state.history:
        if turn["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user-row">
                <div class="avatar av-user">U</div>
                <div class="bub-wrap">
                    <div class="bub-name">You</div>
                    <div class="bub bub-user">{turn['content']}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            is_last    = (turn == st.session_state.history[-1])
            meta       = st.session_state.last_meta or {}
            msg_id     = meta.get("id", "") if is_last else ""
            cur_rating = st.session_state.ratings.get(f"a-{msg_id}", 0) if msg_id else 0

            # Build token info string for bubble footer
            if is_last and meta:
                input_tok  = meta.get("input_tokens", 0)
                output_tok = meta.get("tokens_used",  0)
                elapsed    = meta.get("time_taken_s", 0)
                from_cache = meta.get("from_cache",   False)
                cache_badge = " · ⚡ from cache" if from_cache else ""
                meta_str = (
                    f" · "
                    f"<span style='color:#818cf8'>📥 {input_tok} input tokens</span>"
                    f" · "
                    f"<span style='color:#4ade80'>📤 {output_tok} output tokens</span>"
                    f" · {elapsed}s{cache_badge}"
                )
            else:
                meta_str = ""

            st.markdown(f"""
            <div class="msg-row">
                <div class="avatar av-model">✦</div>
                <div class="bub-wrap">
                    <div class="bub-name">Qwen2.5-0.5B</div>
                    <div class="bub bub-model">{clean_content(turn['content'])}</div>
                    <div class="bub-meta">Qwen2.5-0.5B · {st.session_state.session_id}{meta_str}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Rating buttons ─────────────────────────────────────────────
            if msg_id:
                rc1, rc2, rc3 = st.columns([1, 1, 20])
                with rc1:
                    if st.button("👍", key=f"up_{msg_id}", help="Good response"):
                        new_r = 0 if cur_rating == 1 else 1
                        save_rating(f"a-{msg_id}", st.session_state.session_id, new_r)
                        st.session_state.ratings[f"a-{msg_id}"] = new_r
                        st.rerun()
                with rc2:
                    if st.button("👎", key=f"dn_{msg_id}", help="Bad response"):
                        new_r = 0 if cur_rating == -1 else -1
                        save_rating(f"a-{msg_id}", st.session_state.session_id, new_r)
                        st.session_state.ratings[f"a-{msg_id}"] = new_r
                        st.rerun()

            # ── Continue story button — only on last AI message ────────────
            # Only show if response is long enough to be a story (>100 chars)
            # Avoids showing "Continue" after short replies like "Hello! How can I help?"
            is_story_response = len(turn["content"].strip()) > 100
            if is_last and not st.session_state.generating and is_story_response:
                st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
                st.markdown('<div class="continue-wrap">', unsafe_allow_html=True)
                _, cont_col, _ = st.columns([1, 3, 8])
                with cont_col:
                    if st.button("▶  Continue story", key="continue_btn", help="Ask model to continue from here"):
                        st.session_state.continue_requested = True
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.generating:
        label = f"Processing {len(st.session_state.pending.get('chunks', []))} chunks…" if st.session_state.is_chunking else "Qwen2.5-0.5B is writing…"
        st.markdown(f"""
        <div class="gen-row">
            <div class="avatar av-model">✦</div>
            <div class="bub-wrap">
                <div class="bub-name">{label}</div>
                <div class="typing-bub">
                    <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Input bar
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="input-hint">
    <strong style="color:#6366f1">Enter</strong> to send &nbsp;·&nbsp;
    <strong style="color:#6b7280">Shift+Enter</strong> for new line &nbsp;·&nbsp;
    Smart Chunking handles long inputs automatically
</div>
""", unsafe_allow_html=True)

col_txt, col_btn = st.columns([12, 1])
with col_txt:
    user_input = st.text_area(
        "msg",
        placeholder="Continue a story, paste a long text, or start a new one…",
        height=50,
        key=f"inp_{st.session_state.input_key}",
        label_visibility="collapsed",
    )
with col_btn:
    send = st.button("➤", key="send", disabled=not api_ok)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT ZONE DETECTOR + LIVE COUNTER + CHUNK PANEL
# ══════════════════════════════════════════════════════════════════════════════

# Zone boundaries — easy to change in one place
ZONE_NORMAL   = CHUNK_SIZE          # 0 – 800      → green,  no chunking
ZONE_CHUNKING = 10_000              # 800 – 10000  → yellow, auto chunking
# anything above 10000             → red,    heavy warning

char_count  = len(user_input) if user_input else 0
word_count  = len(user_input.split()) if user_input else 0

# ── Live input token count — calculated locally, no API call needed ───────────
# Rule: 1 token ≈ 4 characters (standard for English text)
# This is instant — updates on every keystroke without any network request
# For exact count the API /count-tokens endpoint is still available
# but calling it on every keystroke is too slow (causes lag)
if char_count > 0:
    input_token_count = max(1, char_count // 4)
else:
    input_token_count = 0

# ── Decide which zone we're in ────────────────────────────────────────────────
if char_count == 0:
    zone = "empty"
elif char_count <= ZONE_NORMAL:
    zone = "normal"
elif char_count <= ZONE_CHUNKING:
    zone = "chunking"
else:
    zone = "danger"

# ── Zone config — colour, label, icon ─────────────────────────────────────────
zone_cfg = {
    "empty":    {"color": "#374151", "bg": "#0f1117",  "border": "#1a1d2e", "icon": "⌨️",  "label": "Start typing…"},
    "normal":   {"color": "#4ade80", "bg": "#0d1f12",  "border": "#14532d", "icon": "✅",  "label": "NORMAL — Ready to send"},
    "chunking": {"color": "#f59e0b", "bg": "#1a1200",  "border": "#78350f", "icon": "🔀",  "label": "CHUNKING ZONE — Will be auto-split"},
    "danger":   {"color": "#ef4444", "bg": "#1f0d0d",  "border": "#7f1d1d", "icon": "⚠️",  "label": "VERY LONG — May take a long time"},
}
cfg = zone_cfg[zone]

# ── Progress bar calculation ───────────────────────────────────────────────────
# Bar fills from 0% to 100% as user types
# 100% = ZONE_CHUNKING (10,000 chars) — beyond that bar stays red at 100%
bar_pct   = min(100, int((char_count / ZONE_CHUNKING) * 100))

# Colour of the progress bar changes with zone
if zone == "normal":
    bar_color = "linear-gradient(90deg, #22c55e, #4ade80)"
elif zone == "chunking":
    bar_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"
else:
    bar_color = "linear-gradient(90deg, #ef4444, #f87171)"

# ── Estimated processing time ──────────────────────────────────────────────────
# Rough estimate: each chunk (~800 chars) takes ~30 seconds on CPU
estimated_chunks = len(split_into_chunks(user_input)) if user_input and char_count > ZONE_NORMAL else 0
estimated_secs   = estimated_chunks * 30   # 30s per chunk on average CPU
if estimated_secs < 60:
    time_estimate = f"~{estimated_secs}s"
else:
    mins = estimated_secs // 60
    secs = estimated_secs % 60
    time_estimate = f"~{mins}m {secs}s" if secs else f"~{mins} min"

# ══════════════════════════════════════════════════════════════════════════════
# RENDER — Zone status bar (always visible)
# ══════════════════════════════════════════════════════════════════════════════

# Pre-compute ALL dynamic values BEFORE building the HTML string
color_normal   = "#4ade80" if zone == "normal"   else "#374151"
color_chunking = "#f59e0b" if zone == "chunking" else "#374151"
color_danger   = "#ef4444" if zone == "danger"   else "#374151"
time_html      = f"&nbsp;·&nbsp; ⏱ Est. time: <span style='color:{cfg['color']};font-weight:600'>{time_estimate}</span>" if estimated_chunks > 0 else ""
bg_col         = cfg["bg"]
border_col     = cfg["border"]
icon_html      = cfg["icon"]
label_html     = cfg["label"]
main_color     = cfg["color"]

# ── Token budget tracker ───────────────────────────────────────────────────────
# input tokens  = what you typed (estimated live)
# output tokens = what the model last generated (from last_meta)
# total         = input + output
# max           = 32,768 (model's hard limit from config.json)
MODEL_MAX       = 32768
last_meta_data  = st.session_state.last_meta or {}
output_tok      = last_meta_data.get("tokens_used", 0)
total_tok       = input_token_count + output_tok
tokens_left     = MODEL_MAX - total_tok
budget_pct      = min(100, int((total_tok / MODEL_MAX) * 100))

# Budget bar colour — green → yellow → red as total approaches limit
if budget_pct < 50:
    budget_bar_color = "linear-gradient(90deg, #22c55e, #4ade80)"
    budget_color     = "#4ade80"
elif budget_pct < 80:
    budget_bar_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"
    budget_color     = "#f59e0b"
else:
    budget_bar_color = "linear-gradient(90deg, #ef4444, #f87171)"
    budget_color     = "#ef4444"

status_html = (
    f'<div style="margin-top:8px; background:{bg_col}; border:1px solid {border_col};'
    f'border-radius:10px; padding:10px 14px;">'

    # ── Row 1: zone label + char/word count ───────────────────────────────
    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">'
    f'<div style="display:flex; align-items:center; gap:8px">'
    f'<span style="font-size:1rem">{icon_html}</span>'
    f'<span style="font-size:0.75rem; font-weight:700; color:{main_color}">{label_html}</span>'
    f'</div>'
    f'<span style="font-size:0.7rem; color:#6b7280">'
    f'<span style="color:{main_color}; font-weight:600">{char_count:,}</span> chars'
    f'&nbsp;·&nbsp;'
    f'<span style="color:{main_color}; font-weight:600">{word_count:,}</span> words'
    f'{time_html}'
    f'</span>'
    f'</div>'

    # ── Row 2: token budget badges ────────────────────────────────────────
    f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap;">'

    # Input token badge
    f'<span style="background:#0d1117; border:1px solid #6366f1; border-radius:6px;'
    f'padding:3px 10px; color:#818cf8; font-weight:600; font-size:0.7rem;">'
    f'&#128229; {input_token_count} input'
    f'</span>'

    f'<span style="color:#374151; font-size:0.8rem;">+</span>'

    # Output token badge
    f'<span style="background:#0d1117; border:1px solid #4ade80; border-radius:6px;'
    f'padding:3px 10px; color:#4ade80; font-weight:600; font-size:0.7rem;">'
    f'&#128228; {output_tok} output'
    f'</span>'

    f'<span style="color:#374151; font-size:0.8rem;">=</span>'

    # Total badge
    f'<span style="background:#0d1117; border:1px solid {budget_color}; border-radius:6px;'
    f'padding:3px 10px; color:{budget_color}; font-weight:700; font-size:0.7rem;">'
    f'&#9889; {total_tok:,} / {MODEL_MAX:,} total'
    f'</span>'

    # Tokens remaining
    f'<span style="font-size:0.65rem; color:#4b5563; margin-left:4px;">'
    f'({tokens_left:,} remaining)'
    f'</span>'
    f'</div>'

    # ── Row 3: input zone progress bar ────────────────────────────────────
    f'<div style="font-size:0.6rem; color:#4b5563; margin-bottom:3px;">Input zone</div>'
    f'<div style="background:#1a1d2e; border-radius:4px; height:5px; overflow:hidden; margin-bottom:6px;">'
    f'<div style="width:{bar_pct}%; height:5px; border-radius:4px; background:{bar_color};"></div>'
    f'</div>'

    # ── Row 4: token budget progress bar ─────────────────────────────────
    f'<div style="font-size:0.6rem; color:#4b5563; margin-bottom:3px;">Token budget ({budget_pct}% used)</div>'
    f'<div style="background:#1a1d2e; border-radius:4px; height:5px; overflow:hidden; margin-bottom:6px;">'
    f'<div style="width:{budget_pct}%; height:5px; border-radius:4px; background:{budget_bar_color};"></div>'
    f'</div>'

    # ── Row 5: zone scale labels ──────────────────────────────────────────
    f'<div style="display:flex; justify-content:space-between; font-size:0.58rem; color:#374151;">'
    f'<span>0</span>'
    f'<span style="color:{color_normal}">Normal (0-{ZONE_NORMAL:,})</span>'
    f'<span style="color:{color_chunking}">Chunking ({ZONE_NORMAL:,}-{ZONE_CHUNKING:,})</span>'
    f'<span style="color:{color_danger}">Very Long ({ZONE_CHUNKING:,}+)</span>'
    f'</div>'

    f'</div>'
)
st.markdown(status_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RENDER — Danger zone warning (only in red zone)
# ══════════════════════════════════════════════════════════════════════════════
if zone == "danger":
    over_by    = char_count - ZONE_CHUNKING
    num_chunks = estimated_chunks
    st.markdown(f"""
    <div style="margin-top:6px; background:#1f0d0d; border:1px solid #7f1d1d;
                border-radius:10px; padding:12px 14px;">
        <div style="font-size:0.78rem; font-weight:700; color:#f87171; margin-bottom:8px">
            ⚠️ Input is very large — {char_count:,} characters ({over_by:,} over recommended limit)
        </div>
        <div style="font-size:0.72rem; color:#9ca3af; line-height:1.8">
            This will be split into <strong style="color:#f87171">{num_chunks} chunks</strong>
            and processed one by one.<br>
            Estimated total processing time:
            <strong style="color:#f87171">{time_estimate}</strong> on CPU.<br><br>
            💡 <strong style="color:#fbbf24">Suggestions to speed it up:</strong><br>
            &nbsp;&nbsp;→ Trim your text to under <strong>{ZONE_CHUNKING:,}</strong> characters<br>
            &nbsp;&nbsp;→ Keep only the most important part of the text<br>
            &nbsp;&nbsp;→ Split it manually and send in separate messages<br>
            &nbsp;&nbsp;→ Lower the Max tokens slider in ⚙ Settings to reduce response size
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RENDER — Chunk preview panel (only in chunking or danger zone)
# ══════════════════════════════════════════════════════════════════════════════
if zone in ("chunking", "danger") and user_input:
    chunks     = split_into_chunks(user_input)
    num_chunks = len(chunks)
    colors     = ["#6366f1", "#8b5cf6", "#a78bfa", "#7c3aed", "#4f46e5"]

    # Only show first 5 chunks in preview — don't flood the screen
    MAX_PREVIEW = 5
    preview_chunks = chunks[:MAX_PREVIEW]
    hidden_chunks  = num_chunks - MAX_PREVIEW

    chunk_blocks = ""
    for i, chunk in enumerate(preview_chunks):
        col        = colors[i % len(colors)]
        wc         = len(chunk.split())
        preview    = chunk[:60].replace("<", "&lt;").replace(">", "&gt;")
        preview    = preview + "…" if len(chunk) > 60 else preview
        storage    = "💾 → RAM → combined at end" if i < num_chunks - 1 else "💾 → RAM → combine ALL → MongoDB"
        chunk_blocks += f"""
        <div style="background:#0f1117; border:1px solid {col};
                    border-left:4px solid {col}; border-radius:8px;
                    padding:8px 12px; margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px">
                <span style="font-size:0.7rem; font-weight:700; color:{col}">CHUNK {i+1} of {num_chunks}</span>
                <span style="font-size:0.62rem; color:#4b5563">{len(chunk):,} chars · {wc} words · {storage}</span>
            </div>
            <div style="font-size:0.7rem; color:#9ca3af; font-style:italic">"{preview}"</div>
        </div>"""

    if hidden_chunks > 0:
        chunk_blocks += f"""
        <div style="text-align:center; font-size:0.68rem; color:#4b5563; padding:6px;">
            + {hidden_chunks} more chunks not shown…
        </div>"""

    # Flow steps
    flow_steps = ""
    for i in range(min(num_chunks, MAX_PREVIEW)):
        col = colors[i % len(colors)]
        flow_steps += f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px">
            <div style="width:20px; height:20px; border-radius:50%; background:{col};
                        display:flex; align-items:center; justify-content:center;
                        font-size:0.6rem; font-weight:700; color:white; flex-shrink:0">{i+1}</div>
            <div style="font-size:0.7rem; color:#9ca3af">
                Chunk {i+1} → model reads it → response saved to
                <span style="color:#a5b4fc">RAM</span>
                {"(context passed to next chunk)" if i < num_chunks-1 else ""}
            </div>
        </div>"""
    if num_chunks > MAX_PREVIEW:
        flow_steps += f"""
        <div style="font-size:0.68rem;color:#4b5563;margin-left:28px">
            … {num_chunks - MAX_PREVIEW} more chunks processed the same way …
        </div>"""
    flow_steps += f"""
    <div style="display:flex; align-items:center; gap:8px; margin-top:4px">
        <div style="width:20px; height:20px; border-radius:50%; background:#4ade80;
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.6rem; font-weight:700; color:#000; flex-shrink:0">✓</div>
        <div style="font-size:0.7rem; color:#4ade80; font-weight:600">
            All {num_chunks} responses joined → ONE combined reply → saved to MongoDB
        </div>
    </div>"""

    st.markdown(f"""
    <div style="margin-top:6px; background:#0a0c10; border:1px solid #2d3148;
                border-radius:12px; padding:14px 16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px">
            <div style="font-size:0.75rem; font-weight:700; color:#a5b4fc">🔀 Chunk Preview</div>
            <div style="font-size:0.65rem; color:#4b5563">
                {char_count:,} chars → {num_chunks} chunks → ~{CHUNK_SIZE} chars each
            </div>
        </div>
        {chunk_blocks}
        <div style="margin-top:10px; padding-top:10px; border-top:1px solid #1a1d2e">
            <div style="font-size:0.62rem; font-weight:700; color:#374151;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px">
                Processing flow:
            </div>
            {flow_steps}
            <div style="margin-top:8px; padding:8px; background:#0d1f12;
                        border:1px solid #14532d; border-radius:6px; font-size:0.68rem; color:#4ade80">
                ✓ Each chunk gets context from the previous one
                &nbsp;·&nbsp; ✓ Story stays coherent across all chunks
                &nbsp;·&nbsp; ✓ Only the final combined result is saved to MongoDB
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Enter key JS ───────────────────────────────────────────────────────────────
st.components.v1.html("""
<script>
(function() {
    const doc = window.parent.document;
    function hook() {
        doc.querySelectorAll('textarea').forEach(ta => {
            if (ta._qwen) return;
            ta._qwen = true;
            ta.addEventListener('keydown', e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                    doc.querySelectorAll('button').forEach(b => {
                        if (b.innerText.trim() === '➤') b.click();
                    });
                }
            });
        });
    }
    hook();
    new MutationObserver(hook).observe(doc.body, { childList:true, subtree:true });
    setTimeout(() => {
        const rows = doc.querySelectorAll('.msg-row, .gen-row');
        if (rows.length) rows[rows.length-1].scrollIntoView({ behavior:'smooth' });
    }, 250);
})();
</script>
""", height=0)


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUE — user clicked "▶ Continue story"
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.continue_requested and not st.session_state.generating:
    continue_prompt = "Continue the story from where you left off. Keep the same style, tone, and characters."

    st.session_state.history.append({"role": "user", "content": continue_prompt})
    st.session_state.generating         = True
    st.session_state.is_chunking        = False
    st.session_state.continue_requested = False
    st.session_state.input_key         += 1
    st.session_state.pending = {
        "prompt":         continue_prompt,
        "chunks":         [continue_prompt],
        "needs_chunking": False,
        "max_tokens":     max_tokens,
        "temperature":    temperature,
        "top_p":          top_p,
        "rep_penalty":    rep_penalty,
    }
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SUBMIT — User clicks send / presses Enter
# ══════════════════════════════════════════════════════════════════════════════
if send and user_input:
    prompt = user_input.strip()

    is_valid, err_title, err_msg = validate_input(prompt)

    if not is_valid:
        icon_map = {
            "Empty Message":         "💬",
            "Too Short":             "📏",
            "Too Long":              "📝",
            "Inappropriate Content": "🚫",
            "English Only":          "🇬🇧",
        }
        st.session_state.popup = {
            "title": err_title,
            "msg":   err_msg,
            "icon":  icon_map.get(err_title, "⚠️"),
        }
        st.rerun()

    else:
        # Check if input needs chunking
        chunks         = split_into_chunks(prompt)
        needs_chunking = len(chunks) > 1

        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.generating  = True
        st.session_state.is_chunking = needs_chunking
        st.session_state.input_key  += 1
        st.session_state.pending = {
            "prompt":         prompt,
            "chunks":         chunks,
            "needs_chunking": needs_chunking,
            "max_tokens":     max_tokens,
            "temperature":    temperature,
            "top_p":          top_p,
            "rep_penalty":    rep_penalty,
            "rag_mode":       rag_mode,
            "rag_top_k":      rag_top_k,
        }
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE — chunked OR streaming
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.generating and st.session_state.pending:
    p = st.session_state.pending

    api_history = [
        {"role": t["role"], "content": t["content"]}
        for t in st.session_state.history[:-1]
    ]

    # ── CHUNKED path — input was too long, split into pieces ──────────────
    if p.get("needs_chunking"):
        chunks            = p["chunks"]
        chunk_placeholder = st.empty()

        # Show live progress for each chunk
        combined      = ""
        working_hist  = api_history.copy()

        for i, chunk in enumerate(chunks):
            # Update progress indicator for each chunk
            chunk_placeholder.markdown(f"""
            <div class="gen-row">
                <div class="avatar av-model">✦</div>
                <div class="bub-wrap">
                    <div class="bub-name">
                        Processing chunk {i+1} of {len(chunks)}
                        — stored in RAM, combining at end…
                    </div>
                    <div style="margin-bottom:6px">
                        <div style="display:flex;gap:4px;margin-top:6px">
                            {"".join([
                                f'<div style="height:4px;flex:1;border-radius:2px;background:{"#6366f1" if j <= i else "#1a1d2e"}"></div>'
                                for j in range(len(chunks))
                            ])}
                        </div>
                    </div>
                    <div class="typing-bub">
                        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Send this chunk
            prompt = chunk if i == 0 else f"Continue based on this next part: {chunk}"
            result = call_generate(
                user_input=prompt,
                history=working_hist,
                settings=p,
                session_id=st.session_state.session_id,
            )

            if result and "error" not in result:
                chunk_response = result["response"]
                combined      += chunk_response + "\n\n"
                # Add to working history so next chunk has context
                working_hist.append({"role": "user",      "content": prompt})
                working_hist.append({"role": "assistant", "content": chunk_response})
            else:
                combined += f"[Part {i+1} could not be processed]\n\n"

        chunk_placeholder.empty()

        if combined.strip():
            st.session_state.history.append({"role": "assistant", "content": combined.strip()})
            title_key = f"title_{st.session_state.session_id}"
            if title_key in st.session_state:
                del st.session_state[title_key]
        else:
            st.session_state.history.append({"role": "assistant", "content": "⚠️ Chunked processing failed."})

    # ── NORMAL streaming path — single chunk, words appear one by one ─────
    else:
        stream_placeholder = st.empty()
        streamed_text      = ""
        last_meta          = {}

        # ── Decide: RAG mode or normal mode ───────────────────────────────
        use_rag   = p.get("rag_mode", False)
        endpoint  = "/rag/ask/stream" if use_rag else "/stream"
        rag_sources = []

        # Build payload — RAG uses query= instead of user_input=
        if use_rag:
            payload = {
                "query":          p["prompt"],
                "history":        api_history,
                "session_id":     st.session_state.session_id,
                "max_new_tokens": p["max_tokens"],
                "temperature":    p["temperature"],
                "top_p":          p["top_p"],
                "rep_penalty":    p["rep_penalty"],
                "top_k":          p.get("rag_top_k", 3),
            }
        else:
            payload = {
                "user_input":         p["prompt"],
                "history":            api_history,
                "session_id":         st.session_state.session_id,
                "max_new_tokens":     p["max_tokens"],
                "temperature":        p["temperature"],
                "top_p":              p["top_p"],
                "repetition_penalty": p["rep_penalty"],
            }

        try:
            with requests.post(
                f"{API_URL}{endpoint}",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    raw = line.decode("utf-8")
                    if not raw.startswith("data: "):
                        continue
                    try:
                        data = json.loads(raw[6:])
                    except:
                        continue

                    # RAG meta event — show sources badge
                    if data.get("type") == "rag_meta":
                        rag_sources = data.get("sources", [])
                        if rag_sources:
                            src_text = ", ".join(rag_sources)
                            stream_placeholder.markdown(
                                f'<div class="msg-row"><div class="avatar av-model">✦</div>'
                                f'<div class="bub-wrap">'
                                f'<div class="bub-name">🔬 RAG — using: <span style="color:#a5b4fc">{src_text}</span></div>'
                                f'<div class="bub bub-model">Searching documents…▌</div>'
                                f'</div></div>',
                                unsafe_allow_html=True
                            )
                        continue

                    if data.get("done"):
                        last_meta = data
                        break

                    streamed_text += data.get("token", "")
                    label = f"🔬 RAG — {', '.join(rag_sources)}" if rag_sources else "Qwen2.5-0.5B is writing…"
                    stream_placeholder.markdown(f"""
                    <div class="msg-row">
                        <div class="avatar av-model">✦</div>
                        <div class="bub-wrap">
                            <div class="bub-name">{label}</div>
                            <div class="bub bub-model">{html_lib.escape(streamed_text)}▌</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Stream error: {e}")

        stream_placeholder.empty()

        if streamed_text:
            # Add source attribution to response if RAG was used
            final_text = streamed_text.strip()
            if use_rag and rag_sources:
                final_text += f"\n\n*Sources: {', '.join(rag_sources)}*"

            st.session_state.history.append({"role": "assistant", "content": final_text})
            st.session_state.last_meta = {
                "time_taken_s": last_meta.get("time_taken_s", 0),
                "tokens_used":  last_meta.get("tokens_used",  0),
                "input_tokens": last_meta.get("input_tokens", 0),
                "from_cache":   last_meta.get("from_cache",   False),
                "id":           last_meta.get("id", ""),
            }
            st.session_state.ai_title_req = st.session_state.session_id
            title_key = f"title_{st.session_state.session_id}"
            if title_key in st.session_state:
                del st.session_state[title_key]
        else:
            st.session_state.history.append({"role": "assistant", "content": "⚠️ No response received."})

    st.session_state.generating  = False
    st.session_state.is_chunking = False
    st.session_state.pending     = None
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# AI TITLE — generate and cache after first message
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.ai_title_req:
    sid       = st.session_state.ai_title_req
    title_key = f"title_{sid}"
    if title_key not in st.session_state:
        ai_title = fetch_ai_title(sid)
        if ai_title:
            st.session_state[title_key] = ai_title
    st.session_state.ai_title_req = None


# ══════════════════════════════════════════════════════════════════════════════
# RAG DOCUMENTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "rag":
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-icon">📚</div>
            <div class="topbar-title">RAG Documents</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Helper functions for RAG API calls ────────────────────────────────
    def rag_fetch_docs():
        try:
            r = requests.get(f"{API_URL}/rag/documents", timeout=5)
            if r.status_code == 200:
                return r.json().get("documents", [])
        except:
            pass
        return None   # None = API error, [] = no docs

    def rag_fetch_stats():
        try:
            r = requests.get(f"{API_URL}/rag/stats", timeout=5)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    def rag_delete_doc(doc_id):
        try:
            r = requests.delete(f"{API_URL}/rag/document/{doc_id}", timeout=5)
            return r.status_code == 200
        except:
            return False

    # ── RAG stats ─────────────────────────────────────────────────────────
    rag_stats = rag_fetch_stats()

    if rag_stats is None:
        st.error("⚠️ Could not connect to RAG API. Make sure FastAPI is running.")
        st.stop()

    if not rag_stats.get("ready", False):
        st.warning("⚠️ RAG not ready. Run: `pip install chromadb sentence-transformers pypdf`")
        st.stop()

    # Stats row
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Documents",     rag_stats.get("total_documents", 0))
    c2.metric("🧩 Total Chunks",  rag_stats.get("total_chunks",    0))
    c3.metric("🤖 Embed Model",   rag_stats.get("embed_model",     "—"))

    st.markdown("---")

    # ── Upload section ────────────────────────────────────────────────────
    st.markdown("### Upload a Document")
    st.markdown(
        "Upload a **PDF**, **TXT**, or **MD** file. "
        "It will be split into chunks, embedded, and stored in ChromaDB. "
        "You can then use **RAG mode** in chat to answer questions based on it."
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md"],
        help="Supported: PDF, TXT, MD files",
    )

    if uploaded:
        if st.button("📥 Index This Document", type="primary"):
            with st.spinner(f"Indexing '{uploaded.name}'... this may take a moment"):
                try:
                    files    = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    response = requests.post(f"{API_URL}/rag/upload", files=files, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(
                            f"✅ **{result['doc_name']}** indexed successfully!  \n"
                            f"Chunks created: **{result['total_chunks']}**  \n"
                            f"Total characters: **{result['total_chars']:,}**"
                        )
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")

    # ── Indexed documents list ─────────────────────────────────────────────
    st.markdown("### Indexed Documents")
    docs = rag_fetch_docs()

    if docs is None:
        st.error("Could not fetch documents.")
    elif len(docs) == 0:
        st.info("No documents indexed yet. Upload a file above to get started.")
    else:
        for doc in docs:
            col_name, col_chunks, col_del = st.columns([5, 2, 1])
            with col_name:
                st.markdown(
                    f'<div style="padding:8px 0; font-size:0.85rem; color:#e5e7eb;">📄 <strong>{doc["doc_name"]}</strong>'
                    f'<span style="color:#6b7280; font-size:0.75rem; margin-left:8px;">ID: {doc["doc_id"]}</span></div>',
                    unsafe_allow_html=True
                )
            with col_chunks:
                st.markdown(
                    f'<div style="padding:8px 0; font-size:0.8rem; color:#9ca3af;">{doc["total_chunks"]} chunks</div>',
                    unsafe_allow_html=True
                )
            with col_del:
                if st.button("🗑️", key=f"del_doc_{doc['doc_id']}", help="Delete this document"):
                    if rag_delete_doc(doc["doc_id"]):
                        st.success(f"Deleted '{doc['doc_name']}'")
                        st.rerun()
                    else:
                        st.error("Delete failed")

    st.markdown("---")

    # ── RAG mode toggle info ───────────────────────────────────────────────
    st.markdown("### Using RAG in Chat")
    st.markdown("""
    After uploading documents, go to **Chat** and enable **🔬 RAG Mode** in the ⚙️ Settings panel.

    **How it works in chat:**
    1. You type your question normally
    2. Before sending to Qwen, the app searches your documents
    3. The most relevant pieces are added to the prompt as context
    4. Qwen answers using YOUR documents — not just its training data

    **Best for:**
    - Asking questions about uploaded PDFs or books
    - Generating stories consistent with your world-building notes
    - Summarising or continuing content from your documents
    """)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "dashboard":
    st.markdown("""
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-icon">📊</div>
            <div class="topbar-title">Stats Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    stats = fetch_stats()
    if not stats:
        st.markdown("""<div class="err-banner">⚠️ Could not load stats. Is FastAPI running?</div>""",
                    unsafe_allow_html=True)
        st.stop()

    t = stats.get("totals", {})
    r = stats.get("response_times", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in [
        (c1, "💬 Total Messages",  t.get("messages", 0)),
        (c2, "🤖 AI Responses",    t.get("ai_responses", 0)),
        (c3, "📁 Sessions",        t.get("sessions", 0)),
        (c4, "🪙 Tokens (approx)", t.get("tokens_approx", 0)),
        (c5, "⏱ Avg Response",    f"{r.get('average_s', 0)}s"),
    ]:
        col.markdown(f"""
        <div style="background:#13151f;border:1px solid #1a1d2e;border-radius:12px;
                    padding:16px;text-align:center;margin-bottom:8px">
            <div style="font-size:0.7rem;color:#6b7280;margin-bottom:6px">{label}</div>
            <div style="font-size:1.4rem;font-weight:700;color:#a5b4fc">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📅 Messages Per Day (Last 7 Days)")
        mpd = stats.get("msgs_per_day", [])
        if mpd:
            labels = [d["date"][-5:] for d in mpd]
            counts = [d["count"] for d in mpd]
            max_c  = max(counts) if max(counts) > 0 else 1
            bars_html = ""
            for lbl, cnt in zip(labels, counts):
                height = int((cnt / max_c) * 80) + 5
                bars_html += f"""
                <div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex:1">
                    <div style="font-size:0.7rem;color:#a5b4fc;font-weight:600">{cnt}</div>
                    <div style="width:100%;height:{height}px;
                                background:linear-gradient(180deg,#6366f1,#8b5cf6);
                                border-radius:4px 4px 0 0"></div>
                    <div style="font-size:0.62rem;color:#6b7280">{lbl}</div>
                </div>"""
            st.markdown(f"""
            <div style="background:#13151f;border:1px solid #1a1d2e;border-radius:12px;padding:20px">
                <div style="display:flex;align-items:flex-end;gap:8px;height:120px">
                    {bars_html}
                </div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🔤 Top Words Used")
        tw = stats.get("top_words", [])
        if tw:
            max_w = tw[0]["count"] if tw else 1
            words_html = ""
            for item in tw[:10]:
                pct = int((item["count"] / max_w) * 100)
                words_html += f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                    <div style="font-size:0.75rem;color:#c9cdd4;width:80px;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{item["word"]}</div>
                    <div style="flex:1;background:#1a1d2e;border-radius:4px;height:8px">
                        <div style="width:{pct}%;background:linear-gradient(90deg,#6366f1,#8b5cf6);
                                    height:8px;border-radius:4px"></div>
                    </div>
                    <div style="font-size:0.65rem;color:#6b7280;width:24px;text-align:right">{item["count"]}</div>
                </div>"""
            st.markdown(f"""
            <div style="background:#13151f;border:1px solid #1a1d2e;border-radius:12px;padding:16px">
                {words_html}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 👍 Rating Summary")
    rat         = stats.get("ratings", {})
    up          = rat.get("thumbs_up", 0)
    down        = rat.get("thumbs_down", 0)
    total_rated = up + down

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.markdown(f"""
    <div style="background:#0d1f12;border:1px solid #14532d;border-radius:12px;
                padding:16px;text-align:center">
        <div style="font-size:1.6rem">👍</div>
        <div style="font-size:1.3rem;font-weight:700;color:#4ade80">{up}</div>
        <div style="font-size:0.7rem;color:#6b7280">Thumbs Up</div>
    </div>""", unsafe_allow_html=True)

    col_r2.markdown(f"""
    <div style="background:#1f0d0d;border:1px solid #7f1d1d;border-radius:12px;
                padding:16px;text-align:center">
        <div style="font-size:1.6rem">👎</div>
        <div style="font-size:1.3rem;font-weight:700;color:#f87171">{down}</div>
        <div style="font-size:0.7rem;color:#6b7280">Thumbs Down</div>
    </div>""", unsafe_allow_html=True)

    satisfaction = round((up / total_rated) * 100) if total_rated > 0 else 0
    col_r3.markdown(f"""
    <div style="background:#13151f;border:1px solid #1a1d2e;border-radius:12px;
                padding:16px;text-align:center">
        <div style="font-size:1.6rem">⭐</div>
        <div style="font-size:1.3rem;font-weight:700;color:#a5b4fc">{satisfaction}%</div>
        <div style="font-size:0.7rem;color:#6b7280">Satisfaction Rate</div>
    </div>""", unsafe_allow_html=True)

    st.stop()