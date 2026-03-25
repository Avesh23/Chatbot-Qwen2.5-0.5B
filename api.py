"""
FastAPI Backend — Qwen2.5-0.5B Story Generator
-----------------------------------------------
Run:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

New endpoints in v3:
  POST /generate           → full response (existing)
  GET  /stream             → SSE streaming response word by word
  POST /rate               → save thumbs up/down rating
  GET  /stats              → dashboard stats (totals, avg time, top words)
  GET  /title/{session_id} → AI-generated session title
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from collections import Counter
from datetime import datetime
import time, uuid, json, re, hashlib, os, tempfile

from model import load_model, generate_response, generate_response_stream, generate_title

# ── RAG engine — import with graceful fallback if not installed ───────────────
try:
    from rag import index_file, get_rag_context, list_documents, delete_document, get_stats as rag_get_stats, evaluate_rag_answer
    RAG_AVAILABLE = True
    print("[API] ✅ RAG engine loaded")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"[API] ⚠ RAG not available: {e}")
    print("[API]   Run: pip install chromadb sentence-transformers pypdf --break-system-packages")


# ══════════════════════════════════════════════════════════════════════════════
# MONGODB
# ══════════════════════════════════════════════════════════════════════════════

MONGO_URI  = "mongodb://localhost:27017"
DB_NAME    = "qwen_story_app"
COL_NAME   = "chat_history"
RATE_COL   = "ratings"
CACHE_COL  = "response_cache"   # ← Option E: new collection for caching responses

def connect_mongodb():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client[DB_NAME]
        print(f"[MongoDB] ✅ Connected → '{DB_NAME}'")
        return client, db[COL_NAME], db[RATE_COL], db[CACHE_COL]
    except ConnectionFailure:
        print("[MongoDB] ❌ Could not connect.")
        return None, None, None, None

mongo_client, messages_col, ratings_col, cache_col = connect_mongodb()


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Qwen2.5-0.5B Story Generator API",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

print("[API] Loading Qwen2.5-0.5B model...")
tokenizer, model = load_model()
print("[API] Model ready.")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role:    str
    content: str

class GenerateRequest(BaseModel):
    user_input:         str                     = Field(..., min_length=1)
    session_id:         Optional[str]           = Field(default=None)
    history:            Optional[list[Message]] = Field(default=[])
    max_new_tokens:     int   = Field(default=100, ge=50,  le=500)  # ← lowered default from 200→100
    temperature:        float = Field(default=0.8, ge=0.1, le=1.5)
    top_p:              float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)

class GenerateResponse(BaseModel):
    id:           str
    session_id:   str
    user_input:   str
    response:     str
    tokens_used:  int
    time_taken_s: float
    model:        str
    saved_to_db:  bool

class RateRequest(BaseModel):
    msg_id:     str   = Field(..., description="The message ID to rate (a-xxxxxxxx)")
    session_id: str
    rating:     int   = Field(..., ge=-1, le=1, description="1=thumbs up, -1=thumbs down, 0=remove")

class MessageRecord(BaseModel):
    id:         str
    session_id: str
    role:       str
    content:    str
    timestamp:  str
    model:      str

class HistoryResponse(BaseModel):
    total:    int
    messages: list[MessageRecord]

class SessionInfo(BaseModel):
    session_id:    str
    message_count: int
    started_at:    str
    last_updated:  str
    title:         Optional[str] = None

class SessionsResponse(BaseModel):
    total:    int
    sessions: list[SessionInfo]

class ClearResponse(BaseModel):
    message: str
    deleted: int


# ══════════════════════════════════════════════════════════════════════════════
# MONGODB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_db_connected() -> bool:
    if mongo_client is None:
        return False
    try:
        mongo_client.admin.command("ping")
        return True
    except:
        return False

def save_message_to_db(session_id, msg_id, role, content, timestamp, time_taken_s=None) -> bool:
    if not is_db_connected() or messages_col is None:
        return False
    try:
        doc = {
            "id":         msg_id,
            "session_id": session_id,
            "role":       role,
            "content":    content,
            "timestamp":  timestamp.isoformat(),
            "model":      "Qwen2.5-0.5B",
        }
        if time_taken_s is not None:
            doc["time_taken_s"] = time_taken_s
        messages_col.insert_one(doc)
        return True
    except Exception as e:
        print(f"[MongoDB] ❌ Save failed: {e}")
        return False

def get_messages_from_db(session_id=None) -> list[dict]:
    if not is_db_connected() or messages_col is None:
        return []
    try:
        query = {"session_id": session_id} if session_id else {}
        return list(messages_col.find(query, {"_id": 0}).sort("timestamp", 1))
    except Exception as e:
        print(f"[MongoDB] ❌ Fetch failed: {e}")
        return []

def get_all_sessions() -> list[dict]:
    if not is_db_connected() or messages_col is None:
        return []
    try:
        pipeline = [
            {"$sort": {"timestamp": 1}},
            {"$group": {
                "_id":           "$session_id",
                "message_count": {"$sum": 1},
                "started_at":    {"$first": "$timestamp"},
                "last_updated":  {"$last":  "$timestamp"},
            }},
            {"$sort": {"last_updated": -1}},
        ]
        return list(messages_col.aggregate(pipeline))
    except Exception as e:
        print(f"[MongoDB] ❌ Sessions failed: {e}")
        return []

def delete_messages_from_db(session_id=None) -> int:
    if not is_db_connected() or messages_col is None:
        return 0
    try:
        query  = {"session_id": session_id} if session_id else {}
        result = messages_col.delete_many(query)
        return result.deleted_count
    except Exception as e:
        print(f"[MongoDB] ❌ Delete failed: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE CACHE HELPERS  (Option E — instant replies for repeated prompts)
# ══════════════════════════════════════════════════════════════════════════════

def make_cache_key(user_input: str, max_new_tokens: int, temperature: float) -> str:
    """
    Create a unique key for a prompt + settings combination.

    How it works:
      We hash the user_input + settings into a short string.
      Same input + same settings = same hash = cache hit → instant reply.
      Different input or different settings = different hash = cache miss → run model.

    We include max_new_tokens and temperature in the key because:
      Same prompt with temp=0.5 vs temp=1.2 should give different cached responses.
      Same prompt with 100 tokens vs 500 tokens are also different.

    We do NOT include session_id or history in the key because:
      We want the same standalone question to hit cache regardless of session.
      Example: "what is a dragon?" should return cached answer every time.
    """
    raw = f"{user_input.strip().lower()}|{max_new_tokens}|{round(temperature, 1)}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached_response(cache_key: str) -> str | None:
    """
    Look up a cache key in MongoDB.
    Returns the cached response string if found, None if not found.

    Cache document structure in MongoDB:
    {
        "cache_key":  "a1b2c3d4e5f6...",   ← MD5 hash of prompt+settings
        "response":   "The knight entered the dark hall...",
        "hit_count":  5,                    ← how many times this was served from cache
        "created_at": "2024-01-15T10:30:00",
        "last_used":  "2024-01-15T11:00:00"
    }
    """
    if cache_col is None:
        return None
    try:
        doc = cache_col.find_one({"cache_key": cache_key})
        if doc:
            # Update hit count and last_used timestamp
            cache_col.update_one(
                {"cache_key": cache_key},
                {"$inc": {"hit_count": 1}, "$set": {"last_used": datetime.now().isoformat()}}
            )
            print(f"[Cache] ✅ HIT — served from cache (hit #{doc.get('hit_count', 1) + 1})")
            return doc["response"]
        return None
    except Exception as e:
        print(f"[Cache] ❌ Lookup failed: {e}")
        return None


def save_to_cache(cache_key: str, response: str) -> None:
    """Save a new response to the cache for future instant retrieval."""
    if cache_col is None:
        return
    try:
        cache_col.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key":  cache_key,
                "response":   response,
                "hit_count":  0,
                "created_at": datetime.now().isoformat(),
                "last_used":  datetime.now().isoformat(),
            }},
            upsert=True,
        )
        print(f"[Cache] 💾 Saved new response to cache")
    except Exception as e:
        print(f"[Cache] ❌ Save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — EXISTING
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "message": "Qwen2.5-0.5B API v3 running",
        "mongodb": "connected" if is_db_connected() else "disconnected",
        "docs":    "http://localhost:8000/docs",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — TOKEN COUNTER  🔢
# ══════════════════════════════════════════════════════════════════════════════

class CountTokensRequest(BaseModel):
    text: str

@app.post("/count-tokens", tags=["Tokens"])
def count_tokens(req: CountTokensRequest):
    """
    Returns the EXACT token count for any text using the real tokenizer.

    This is how the model actually counts — not an estimate.
    The tokenizer splits text into subword pieces:
      "unbelievable" → ["un", "believe", "able"] → 3 tokens
      "hello"        → ["hello"]                 → 1 token

    Used by the UI to show live input token count as user types,
    and to show exact output token count after generation.
    """
    if not req.text:
        return {"token_count": 0, "char_count": 0, "word_count": 0}

    # tokenizer() returns a dict with "input_ids" — a tensor of token IDs
    # .shape[1] gives the number of tokens (columns in the 2D tensor)
    token_ids   = tokenizer(req.text, return_tensors="pt")["input_ids"]
    token_count = int(token_ids.shape[1])
    char_count  = len(req.text)
    word_count  = len(req.text.split())

    return {
        "token_count": token_count,
        "char_count":  char_count,
        "word_count":  word_count,
    }


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
def generate(req: GenerateRequest):
    """Full response generation (non-streaming) with cache support."""
    if not req.user_input.strip():
        raise HTTPException(400, "user_input cannot be empty")

    session_id    = req.session_id or f"sess-{str(uuid.uuid4())[:8]}"
    msg_id        = str(uuid.uuid4())[:8]
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    # ── Check cache first ──────────────────────────────────────────────────
    # If the exact same prompt was asked before with same settings
    # → return the cached response instantly (0 seconds!)
    cache_key     = make_cache_key(req.user_input, req.max_new_tokens, req.temperature)
    cached        = get_cached_response(cache_key)

    start = time.time()

    if cached:
        # Cache hit → return immediately, no model needed
        response_text = cached
        elapsed       = round(time.time() - start, 4)
        print(f"[Cache] ⚡ Instant reply from cache in {elapsed}s")
    else:
        # Cache miss → run model
        try:
            response_text = generate_response(
                tokenizer=tokenizer, model=model,
                history=history_dicts, user_input=req.user_input,
                max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                top_p=req.top_p, repetition_penalty=req.repetition_penalty,
            )
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}")
        elapsed = round(time.time() - start, 2)
        # Save to cache for next time
        save_to_cache(cache_key, response_text)

    # Real token count using tokenizer (not estimate)
    input_tokens  = int(tokenizer(req.user_input,  return_tensors="pt")["input_ids"].shape[1])
    output_tokens = int(tokenizer(response_text,   return_tensors="pt")["input_ids"].shape[1])
    now           = datetime.now()

    saved_user = save_message_to_db(session_id, f"u-{msg_id}", "user",      req.user_input,  now)
    saved_asst = save_message_to_db(session_id, f"a-{msg_id}", "assistant", response_text,   datetime.now(), elapsed)

    return GenerateResponse(
        id=msg_id, session_id=session_id,
        user_input=req.user_input, response=response_text,
        tokens_used=output_tokens, time_taken_s=elapsed,
        model="Qwen2.5-0.5B", saved_to_db=saved_user and saved_asst,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — STREAMING  ⚡
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/stream", tags=["Generation"])
def stream(req: GenerateRequest):
    """
    SSE streaming endpoint with cache support.
    Cache hit  → replays the cached response word by word (still feels like streaming!)
    Cache miss → runs model, streams tokens, saves to cache for next time.
    """
    if not req.user_input.strip():
        raise HTTPException(400, "user_input cannot be empty")

    session_id    = req.session_id or f"sess-{str(uuid.uuid4())[:8]}"
    msg_id        = str(uuid.uuid4())[:8]
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    # Check cache before starting generation
    cache_key = make_cache_key(req.user_input, req.max_new_tokens, req.temperature)
    cached    = get_cached_response(cache_key)

    def event_generator():
        start = time.time()

        # Save user message immediately
        save_message_to_db(session_id, f"u-{msg_id}", "user", req.user_input, datetime.now())

        if cached:
            # ── Cache hit: replay cached response word by word ─────────────
            # Split into words and send each one
            # Still feels like streaming to the user but is near-instant
            print(f"[Cache] ⚡ Streaming from cache")
            words = cached.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                payload = json.dumps({"token": token, "done": False})
                yield f"data: {payload}\n\n"

            elapsed = round(time.time() - start, 3)
            save_message_to_db(session_id, f"a-{msg_id}", "assistant", cached.strip(), datetime.now(), elapsed)
            output_tokens = int(tokenizer(cached, return_tensors="pt")["input_ids"].shape[1])
            input_tokens  = int(tokenizer(req.user_input, return_tensors="pt")["input_ids"].shape[1])
            done_payload = json.dumps({
                "token": "", "done": True,
                "session_id":    session_id,
                "id":            msg_id,
                "time_taken_s":  elapsed,
                "tokens_used":   output_tokens,
                "input_tokens":  input_tokens,
                "from_cache":    True,
            })
            yield f"data: {done_payload}\n\n"

        else:
            # ── Cache miss: run model, stream tokens, save to cache ────────
            full_response = ""
            try:
                for token in generate_response_stream(
                    tokenizer=tokenizer, model=model,
                    history=history_dicts, user_input=req.user_input,
                    max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                    top_p=req.top_p, repetition_penalty=req.repetition_penalty,
                ):
                    full_response += token
                    payload = json.dumps({"token": token, "done": False})
                    yield f"data: {payload}\n\n"

            except Exception as e:
                err = json.dumps({"token": f"[Error: {e}]", "done": True})
                yield f"data: {err}\n\n"
                return

            elapsed = round(time.time() - start, 2)

            # Save to MongoDB and cache
            save_message_to_db(session_id, f"a-{msg_id}", "assistant", full_response.strip(), datetime.now(), elapsed)
            save_to_cache(cache_key, full_response.strip())

            output_tokens = int(tokenizer(full_response, return_tensors="pt")["input_ids"].shape[1])
            input_tokens  = int(tokenizer(req.user_input, return_tensors="pt")["input_ids"].shape[1])
            done_payload = json.dumps({
                "token": "", "done": True,
                "session_id":    session_id,
                "id":            msg_id,
                "time_taken_s":  elapsed,
                "tokens_used":   output_tokens,
                "input_tokens":  input_tokens,
                "from_cache":    False,
            })
            yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — RATING  👍 👎
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/rate", tags=["Ratings"])
def rate_message(req: RateRequest):
    """
    Save a thumbs up (1) or thumbs down (-1) rating for an AI response.
    Ratings are stored in the 'ratings' MongoDB collection.
    """
    if not is_db_connected() or ratings_col is None:
        raise HTTPException(503, "MongoDB not connected")

    # Upsert — update if already rated, insert if new
    ratings_col.update_one(
        {"msg_id": req.msg_id},
        {"$set": {
            "msg_id":     req.msg_id,
            "session_id": req.session_id,
            "rating":     req.rating,
            "timestamp":  datetime.now().isoformat(),
        }},
        upsert=True,
    )
    label = "👍 Thumbs up" if req.rating == 1 else "👎 Thumbs down" if req.rating == -1 else "Removed"
    return {"message": f"Rating saved: {label}", "msg_id": req.msg_id}


@app.get("/ratings/{session_id}", tags=["Ratings"])
def get_session_ratings(session_id: str):
    """Get all ratings for a session so UI can show which messages were rated."""
    if not is_db_connected() or ratings_col is None:
        raise HTTPException(503, "MongoDB not connected")
    docs = list(ratings_col.find({"session_id": session_id}, {"_id": 0}))
    return {"session_id": session_id, "ratings": docs}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — AI TITLE  🏷️
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/title/{session_id}", tags=["Sessions"])
def get_session_title(session_id: str):
    """
    Generate an AI title for a session using the first user message.
    Much better than just showing the raw first message.
    """
    docs = get_messages_from_db(session_id=session_id)
    if not docs:
        raise HTTPException(404, f"No messages for session '{session_id}'")

    # Find first user message
    first_msg = next((d["content"] for d in docs if d["role"] == "user"), None)
    if not first_msg:
        raise HTTPException(404, "No user message found")

    title = generate_title(tokenizer, model, first_msg)
    return {"session_id": session_id, "title": title}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — STATS DASHBOARD  📊
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/stats", tags=["Dashboard"])
def get_stats():
    """
    Returns all stats needed for the dashboard:
    - Total messages, sessions, tokens
    - Average response time
    - Messages per day (last 7 days)
    - Top 15 most used words
    - Rating summary (thumbs up vs down)
    """
    if not is_db_connected():
        raise HTTPException(503, "MongoDB not connected")

    all_docs     = get_messages_from_db()
    user_msgs    = [d for d in all_docs if d["role"] == "user"]
    asst_msgs    = [d for d in all_docs if d["role"] == "assistant"]
    all_sessions = get_all_sessions()

    # ── Response times ────────────────────────────────────────────────────
    times = [
        d.get("time_taken_s", 0)
        for d in asst_msgs
        if d.get("time_taken_s")
    ]
    avg_time = round(sum(times) / len(times), 2) if times else 0
    min_time = round(min(times), 2) if times else 0
    max_time = round(max(times), 2) if times else 0

    # ── Token estimate ────────────────────────────────────────────────────
    total_tokens = sum(len(d["content"]) // 4 for d in asst_msgs)

    # ── Messages per day (last 7 days) ────────────────────────────────────
    from datetime import timedelta
    today      = datetime.now().date()
    day_counts = {}
    for i in range(6, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        day_counts[day] = 0

    for d in all_docs:
        try:
            ts  = datetime.fromisoformat(d["timestamp"]).date().isoformat()
            if ts in day_counts:
                day_counts[ts] += 1
        except:
            pass

    msgs_per_day = [{"date": k, "count": v} for k, v in day_counts.items()]

    # ── Top words ─────────────────────────────────────────────────────────
    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","is","it","its","i","you","he","she","they","we",
        "was","be","are","this","that","as","by","from","not","have",
        "had","has","do","did","will","would","can","could","my","your",
        "his","her","their","our","just","so","if","about","up","out",
        "what","when","how","there","which","who","him","them","been",
    }
    all_text  = " ".join(d["content"] for d in all_docs)
    words     = re.findall(r"\b[a-zA-Z]{4,}\b", all_text.lower())
    filtered  = [w for w in words if w not in STOPWORDS]
    top_words = [{"word": w, "count": c} for w, c in Counter(filtered).most_common(15)]

    # ── Ratings summary ───────────────────────────────────────────────────
    thumbs_up   = 0
    thumbs_down = 0
    if ratings_col is not None:
        thumbs_up   = ratings_col.count_documents({"rating":  1})
        thumbs_down = ratings_col.count_documents({"rating": -1})

    # ── Cache summary ─────────────────────────────────────────────────────
    cache_entries = 0
    total_hits    = 0
    if cache_col is not None:
        cache_entries = cache_col.count_documents({})
        agg = list(cache_col.aggregate([{"$group": {"_id": None, "total": {"$sum": "$hit_count"}}}]))
        total_hits = agg[0]["total"] if agg else 0

    return {
        "totals": {
            "messages":       len(all_docs),
            "user_messages":  len(user_msgs),
            "ai_responses":   len(asst_msgs),
            "sessions":       len(all_sessions),
            "tokens_approx":  total_tokens,
        },
        "response_times": {
            "average_s": avg_time,
            "min_s":     min_time,
            "max_s":     max_time,
        },
        "msgs_per_day": msgs_per_day,
        "top_words":    top_words,
        "ratings": {
            "thumbs_up":   thumbs_up,
            "thumbs_down": thumbs_down,
        },
        "cache": {
            "cached_responses": cache_entries,
            "total_cache_hits": total_hits,
            "time_saved_approx": f"~{total_hits * 20}s",  # rough estimate
        },
    }


@app.post("/cache/clear", tags=["Cache"])
def clear_cache():
    """Clear all cached responses. Useful after model changes or for testing."""
    if cache_col is None:
        raise HTTPException(503, "MongoDB not connected")
    result = cache_col.delete_many({})
    return {"message": "Cache cleared", "deleted": result.deleted_count}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — HISTORY (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/history", response_model=HistoryResponse, tags=["History"])
def get_all_history():
    if not is_db_connected():
        raise HTTPException(503, "MongoDB not connected")
    docs = get_messages_from_db()
    return HistoryResponse(total=len(docs), messages=[MessageRecord(**d) for d in docs])


@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["History"])
def get_session_history(session_id: str):
    if not is_db_connected():
        raise HTTPException(503, "MongoDB not connected")
    docs = get_messages_from_db(session_id=session_id)
    if not docs:
        raise HTTPException(404, f"No messages for session '{session_id}'")
    return HistoryResponse(total=len(docs), messages=[MessageRecord(**d) for d in docs])


@app.get("/sessions", response_model=SessionsResponse, tags=["History"])
def list_sessions():
    if not is_db_connected():
        raise HTTPException(503, "MongoDB not connected")
    raw      = get_all_sessions()
    sessions = [
        SessionInfo(
            session_id=s["_id"], message_count=s["message_count"],
            started_at=s["started_at"], last_updated=s["last_updated"],
        )
        for s in raw
    ]
    return SessionsResponse(total=len(sessions), sessions=sessions)


@app.post("/history/clear", response_model=ClearResponse, tags=["History"])
def clear_all_history():
    if not is_db_connected():
        raise HTTPException(503, "MongoDB not connected")
    deleted = delete_messages_from_db()
    return ClearResponse(message="All history cleared", deleted=deleted)


@app.delete("/history/{session_id}", response_model=ClearResponse, tags=["History"])
def delete_session_route(session_id: str):
    if not is_db_connected():
        raise HTTPException(503, "MongoDB not connected")
    deleted = delete_messages_from_db(session_id=session_id)
    if deleted == 0:
        raise HTTPException(404, f"No messages for session '{session_id}'")
    return ClearResponse(message=f"Session '{session_id}' deleted", deleted=deleted)


# ══════════════════════════════════════════════════════════════════════════════
# RAG ROUTES
# ══════════════════════════════════════════════════════════════════════════════

class RagAskRequest(BaseModel):
    query:       str            = Field(..., min_length=1)
    session_id:  Optional[str]  = Field(default=None)
    history:     Optional[list[Message]] = Field(default=[])
    doc_id:      Optional[str]  = Field(default=None,  description="Search only in this document. None = search all.")
    top_k:       int            = Field(default=3, ge=1, le=10, description="How many chunks to retrieve")
    max_new_tokens: int         = Field(default=200, ge=50, le=500)
    temperature: float          = Field(default=0.8, ge=0.1, le=1.5)


@app.post("/rag/upload", tags=["RAG"])
async def rag_upload(file: UploadFile = File(...)):
    """
    Upload a document (PDF or TXT) and index it into ChromaDB.

    What happens:
      1. File is saved temporarily to disk
      2. Text is extracted (pypdf for PDF, plain read for TXT)
      3. Text is split into overlapping chunks (~1500 chars each)
      4. Each chunk is embedded using sentence-transformers
      5. Embeddings + text saved to ChromaDB on disk
      6. File is deleted from disk (only vectors remain)

    After this, the document is permanently searchable
    until you delete it via DELETE /rag/document/{doc_id}
    """
    if not RAG_AVAILABLE:
        raise HTTPException(503, "RAG not available. Install: chromadb sentence-transformers pypdf")

    # Validate file type
    allowed = {".pdf", ".txt", ".md"}
    ext     = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type '{ext}' not supported. Allowed: {allowed}")

    # Save uploaded file to a temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Index the document
        result = index_file(tmp_path, doc_name=file.filename)

    except Exception as e:
        raise HTTPException(500, f"Failed to index document: {e}")
    finally:
        # Always delete temp file after indexing
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if "error" in result:
        raise HTTPException(500, result["error"])

    return {
        "message":      f"Document '{file.filename}' indexed successfully",
        "doc_id":       result["doc_id"],
        "doc_name":     result["doc_name"],
        "total_chunks": result["total_chunks"],
        "total_chars":  result["total_chars"],
    }


@app.post("/rag/ask", tags=["RAG"])
def rag_ask(req: RagAskRequest):
    """
    Ask a question using RAG — retrieves relevant document chunks
    then feeds them to Qwen as context.

    Flow:
      1. Embed the query using sentence-transformers
      2. Search ChromaDB for top_k most similar chunks
      3. Build augmented prompt:
           "Use this context: [chunks]... Answer: [query]"
      4. Send augmented prompt to Qwen
      5. Return response + which sources were used

    If no documents are indexed → falls back to normal generation
    """
    if not RAG_AVAILABLE:
        raise HTTPException(503, "RAG not available")

    session_id    = req.session_id or f"sess-{str(uuid.uuid4())[:8]}"
    msg_id        = str(uuid.uuid4())[:8]
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    # Step 1+2+3: retrieve chunks + build augmented prompt
    rag_result = get_rag_context(req.query, top_k=req.top_k, doc_id=req.doc_id)

    augmented_prompt = rag_result["augmented_prompt"]
    has_context      = rag_result["has_context"]
    sources          = rag_result["sources"]
    chunks_used      = rag_result["chunks"]

    # Step 4: generate response using augmented prompt
    start = time.time()
    try:
        # Pass augmented_prompt as user_input — it already contains the context
        response_text = generate_response(
            tokenizer      = tokenizer,
            model          = model,
            history        = history_dicts,
            user_input     = augmented_prompt,
            max_new_tokens = req.max_new_tokens,
            temperature    = req.temperature,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    elapsed = round(time.time() - start, 2)

    # Save to MongoDB
    save_message_to_db(session_id, f"u-{msg_id}", "user",      req.query,      datetime.now())
    save_message_to_db(session_id, f"a-{msg_id}", "assistant", response_text,  datetime.now(), elapsed)

    # ── Auto-evaluate the answer and print to terminal ─────────────────────
    evaluation = {}
    if has_context:
        evaluation = evaluate_rag_answer(
            query   = req.query,
            answer  = response_text,
            chunks  = chunks_used,
        )

    return {
        "id":           msg_id,
        "session_id":   session_id,
        "query":        req.query,
        "response":     response_text,
        "has_context":  has_context,
        "sources":      sources,
        "chunks_used":  [{"doc_name": c["doc_name"], "similarity": c["similarity"], "preview": c["text"][:100]} for c in chunks_used],
        "time_taken_s": elapsed,
        "model":        "Qwen2.5-0.5B + RAG",
        "evaluation":   evaluation,
    }


@app.post("/rag/ask/stream", tags=["RAG"])
def rag_ask_stream(req: RagAskRequest):
    """
    Streaming version of /rag/ask — returns tokens word by word via SSE.
    Same RAG pipeline but streams the response as it generates.
    """
    if not RAG_AVAILABLE:
        raise HTTPException(503, "RAG not available")

    session_id    = req.session_id or f"sess-{str(uuid.uuid4())[:8]}"
    msg_id        = str(uuid.uuid4())[:8]
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    # Retrieve + build augmented prompt BEFORE streaming starts
    rag_result       = get_rag_context(req.query, top_k=req.top_k, doc_id=req.doc_id)
    augmented_prompt = rag_result["augmented_prompt"]
    sources          = rag_result["sources"]
    has_context      = rag_result["has_context"]

    def event_generator():
        full_response = ""
        start         = time.time()

        # Send RAG metadata as first SSE event so UI knows sources
        meta_payload = json.dumps({
            "type":        "rag_meta",
            "has_context": has_context,
            "sources":     sources,
            "done":        False,
        })
        yield f"data: {meta_payload}\n\n"

        save_message_to_db(session_id, f"u-{msg_id}", "user", req.query, datetime.now())

        try:
            for token in generate_response_stream(
                tokenizer      = tokenizer,
                model          = model,
                history        = history_dicts,
                user_input     = augmented_prompt,
                max_new_tokens = req.max_new_tokens,
                temperature    = req.temperature,
            ):
                full_response += token
                payload = json.dumps({"token": token, "done": False, "type": "token"})
                yield f"data: {payload}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'token': f'[Error: {e}]', 'done': True})}\n\n"
            return

        elapsed = round(time.time() - start, 2)
        save_message_to_db(session_id, f"a-{msg_id}", "assistant", full_response.strip(), datetime.now(), elapsed)

        done_payload = json.dumps({
            "token":        "",
            "done":         True,
            "type":         "done",
            "session_id":   session_id,
            "id":           msg_id,
            "time_taken_s": elapsed,
            "has_context":  has_context,
            "sources":      sources,
        })
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/rag/documents", tags=["RAG"])
def rag_list_documents():
    """List all indexed documents in ChromaDB."""
    if not RAG_AVAILABLE:
        raise HTTPException(503, "RAG not available")
    docs = list_documents()
    return {"total": len(docs), "documents": docs}


@app.delete("/rag/document/{doc_id}", tags=["RAG"])
def rag_delete_document(doc_id: str):
    """Delete a document and all its chunks from ChromaDB."""
    if not RAG_AVAILABLE:
        raise HTTPException(503, "RAG not available")
    deleted = delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    return {"message": f"Deleted {deleted} chunks for doc '{doc_id}'", "deleted": deleted}


@app.get("/rag/stats", tags=["RAG"])
def rag_stats():
    """RAG system stats — document count, chunk count, model info."""
    if not RAG_AVAILABLE:
        return {"ready": False, "message": "RAG not installed"}
    return rag_get_stats()


class EvaluateRequest(BaseModel):
    query:   str = Field(..., min_length=1)
    answer:  str = Field(..., min_length=1)
    doc_id:  Optional[str] = Field(default=None)
    top_k:   int = Field(default=3, ge=1, le=10)


@app.post("/rag/evaluate", tags=["RAG"])
def rag_evaluate(req: EvaluateRequest):
    """
    Evaluate whether a RAG answer is correct and grounded.

    Pass in the question + the answer you got.
    Returns 4 scores:

    1. Context Relevance   — were the right chunks retrieved? (30% weight)
    2. Answer Groundedness — is the answer from the chunks?   (35% weight)
    3. Answer Completeness — did model use chunks well?       (15% weight)
    4. Query Coverage      — did answer address the question? (20% weight)

    Overall score + grade (A/B/C/D) printed to terminal too.

    Example use:
      POST /rag/evaluate
      {
        "query":  "Who is Atlas Corrigan?",
        "answer": "Atlas is a homeless boy Lily meets at age 15..."
      }
    """
    if not RAG_AVAILABLE:
        raise HTTPException(503, "RAG not available")

    # Retrieve the same chunks that would have been used
    rag_result  = get_rag_context(req.query, top_k=req.top_k, doc_id=req.doc_id)
    chunks_used = rag_result["chunks"]

    if not chunks_used:
        return {
            "error":   "No chunks found — upload a document first",
            "overall_score": 0,
            "grade":   "F",
            "verdict": "❌ No document context available to evaluate against",
        }

    evaluation = evaluate_rag_answer(
        query  = req.query,
        answer = req.answer,
        chunks = chunks_used,
    )

    return evaluation