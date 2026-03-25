"""
rag.py — RAG Engine for Qwen2.5-0.5B Story Generator
======================================================
RAG = Retrieval Augmented Generation

What this file does:
  1. Takes documents (PDF / TXT) uploaded by user
  2. Splits them into small chunks (~300 words each)
  3. Converts each chunk into a vector (embedding) using sentence-transformers
  4. Stores vectors + text in ChromaDB (local vector database on disk)
  5. When user asks a question → finds most similar chunks → gives to Qwen

Install requirements:
  pip install chromadb sentence-transformers pypdf --break-system-packages

How embeddings work (simple explanation):
  "The dragon lives in the north"  → [0.23, -0.41, 0.88, ...]  384 numbers
  "Dragon habitat northern region" → [0.21, -0.39, 0.90, ...]  very similar!
  "How to bake a cake"             → [0.91,  0.12, -0.33, ...] very different

  Similar meaning = similar numbers = found together in search
"""

import os
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Where ChromaDB stores vectors on disk — persists between restarts
CHROMA_DIR     = "./chroma_db"

# Embedding model — runs 100% locally on CPU, no API key needed
# all-MiniLM-L6-v2:
#   - 80MB download (once)
#   - 384-dimensional vectors
#   - Very fast on CPU (~10ms per chunk)
#   - Great quality for semantic search
EMBED_MODEL    = "all-MiniLM-L6-v2"

# How big each chunk is (in characters)
# ~300 words = ~1500 chars — enough context but not too long
CHUNK_SIZE     = 1500

# How much chunks overlap — so a sentence cut in half still gets found
# Example: chunk1 ends at char 1500, chunk2 starts at char 1200
# The 300 char overlap means no context is lost at boundaries
CHUNK_OVERLAP  = 200

# How many similar chunks to retrieve per query
TOP_K          = 3

# ChromaDB collection name — like a "table" for your documents
COLLECTION     = "rag_documents"


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISE — Load embedding model + ChromaDB (once at startup)
# ══════════════════════════════════════════════════════════════════════════════

print("[RAG] Loading sentence-transformer embedding model...")
try:
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"[RAG] ✅ Embedding model loaded: {EMBED_MODEL}")
except Exception as e:
    print(f"[RAG] ❌ Failed to load embedding model: {e}")
    print("[RAG] Run: pip install sentence-transformers --break-system-packages")
    embedder = None

print("[RAG] Connecting to ChromaDB...")
try:
    # PersistentClient saves everything to disk at CHROMA_DIR
    # So your documents survive app restarts
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)  # disable telemetry
    )

    # Get or create the collection
    # Collection = like a table in a database
    # Each row = one chunk + its embedding vector
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )
    print(f"[RAG] ✅ ChromaDB ready at '{CHROMA_DIR}' | Collection: '{COLLECTION}'")
    print(f"[RAG]    Documents in DB: {collection.count()}")
except Exception as e:
    print(f"[RAG] ❌ ChromaDB failed: {e}")
    chroma_client = None
    collection    = None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CHUNKING
# Split a long document into overlapping pieces
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.

    Why overlapping?
      Imagine a sentence: "The dragon, who had lived in the cave for 300 years..."
      If we split at exactly 1500 chars, this sentence might be cut in half.
      With overlap, the second chunk starts 200 chars earlier,
      so the full sentence appears in both chunks — guaranteed to be found.

    Example with chunk_size=20, overlap=5:
      Text:   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      Chunk1: "ABCDEFGHIJKLMNOPQRST"       (0 to 20)
      Chunk2: "PQRSTUVWXYZ"                (15 to 26)  ← starts 5 back
    """
    chunks = []
    start  = 0
    text   = text.strip()

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward by chunk_size - overlap
        # This creates the overlap between consecutive chunks
        start += chunk_size - overlap

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DOCUMENT LOADING
# Read text from different file types
# ══════════════════════════════════════════════════════════════════════════════

def load_document(file_path: str) -> str:
    """
    Read a document file and return its text content.
    Supports: .txt, .pdf, .md

    For PDF: uses pypdf to extract text from all pages
    For TXT/MD: reads as plain text
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text   = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
            print(f"[RAG] PDF loaded: {len(reader.pages)} pages, {len(text)} chars")
            return text
        except ImportError:
            raise ImportError("pypdf not installed. Run: pip install pypdf --break-system-packages")
        except Exception as e:
            raise Exception(f"Failed to read PDF: {e}")

    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        print(f"[RAG] Text file loaded: {len(text)} chars")
        return text

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EMBEDDING + STORING
# Convert chunks to vectors and save in ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

def index_document(
    text: str,
    doc_name: str,
    doc_id:   Optional[str] = None,
) -> dict:
    """
    Full pipeline: text → chunks → embeddings → ChromaDB

    Steps:
      1. Split text into overlapping chunks
      2. Use sentence-transformer to embed each chunk
         (converts text to list of 384 numbers)
      3. Store chunks + embeddings + metadata in ChromaDB

    Returns summary dict with stats.

    How ChromaDB stores data:
      Each entry has:
        id        → unique string (doc_id + chunk number)
        embedding → [0.23, -0.41, 0.88, ...] 384 numbers
        document  → "The dragon lives in the north mountains..."
        metadata  → {"doc_name": "story.pdf", "chunk_index": 0, ...}
    """
    if embedder is None or collection is None:
        return {"error": "RAG not initialised — check embedder and ChromaDB"}

    doc_id  = doc_id or str(uuid.uuid4())[:8]
    chunks  = chunk_text(text)

    if not chunks:
        return {"error": "No text found in document"}

    # ── TERMINAL: Print all chunks on upload ──────────────────────────────
    print()
    print("=" * 70)
    print("  [RAG] INDEXING DOCUMENT: " + doc_name)
    print("  Doc ID      : " + doc_id)
    print("  Total chars : " + str(len(text)))
    print("  Total chunks: " + str(len(chunks)))
    print("  Chunk size  : " + str(CHUNK_SIZE) + " chars  |  Overlap: " + str(CHUNK_OVERLAP) + " chars")
    print("=" * 70)

    for i, chunk in enumerate(chunks):
        word_count  = len(chunk.split())
        preview     = chunk[:200].replace("\n", " ").strip()
        preview_str = preview + "..." if len(chunk) > 200 else preview

        print()
        print("  ┌── CHUNK " + str(i+1) + " of " + str(len(chunks)) + " ──────────────────────────────────────────────")
        print("  │  Chars : " + str(len(chunk)) + "  |  Words : " + str(word_count))
        print("  │  ID    : " + doc_id + "_chunk_" + str(i))
        print("  │")
        print("  │  CONTENT PREVIEW:")

        words = preview_str.split()
        line  = "  │    "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "  │    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        print("  └" + "─" * 68)

    print()
    print("  [RAG] Embedding " + str(len(chunks)) + " chunks using " + EMBED_MODEL + "...")
    print()

    # Embed all chunks at once (batch processing = faster)
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

    # Build lists for ChromaDB batch insert
    ids        = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas  = [
        {
            "doc_name":     doc_name,
            "doc_id":       doc_id,
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "char_count":   len(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]

    # Insert everything into ChromaDB in one batch
    collection.add(
        ids        = ids,
        embeddings = embeddings,
        documents  = chunks,
        metadatas  = metadatas,
    )

    print()
    print("=" * 70)
    print(f"  [RAG] ✅ INDEXING COMPLETE")
    print(f"  Document  : '{doc_name}'")
    print(f"  Chunks    : {len(chunks)} stored in ChromaDB")
    print(f"  Total DB  : {collection.count()} chunks across all documents")
    print("=" * 70)
    print()

    return {
        "doc_id":       doc_id,
        "doc_name":     doc_name,
        "total_chunks": len(chunks),
        "total_chars":  len(text),
        "chunk_size":   CHUNK_SIZE,
        "overlap":      CHUNK_OVERLAP,
    }


def index_file(file_path: str, doc_name: Optional[str] = None) -> dict:
    """
    Convenience function: file path → load → index
    Used by the /upload endpoint in api.py
    """
    doc_name = doc_name or os.path.basename(file_path)
    text     = load_document(file_path)
    return index_document(text, doc_name)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RETRIEVAL
# Search ChromaDB for chunks similar to the user's question
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_similar(query: str, top_k: int = TOP_K, doc_id: Optional[str] = None) -> list[dict]:
    """
    Find the most relevant chunks for a given query.

    How it works:
      1. Embed the query using the same embedding model
      2. ChromaDB compares query vector against all stored chunk vectors
      3. Returns top_k chunks sorted by cosine similarity (most similar first)

    Cosine similarity score:
      1.0 = identical meaning
      0.8+ = very similar
      0.5  = somewhat related
      0.0  = completely different

    Optional doc_id filter: only search within one specific document
    """
    if embedder is None or collection is None:
        return []

    if collection.count() == 0:
        return []

    # Embed the query into a vector
    query_embedding = embedder.encode([query]).tolist()

    # Build optional filter — search only in a specific document
    where = {"doc_id": doc_id} if doc_id else None

    # Query ChromaDB — finds most similar chunks
    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = min(top_k, collection.count()),
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )

    # Format results into clean list of dicts
    chunks = []
    for i in range(len(results["ids"][0])):
        distance  = results["distances"][0][i]
        # Convert distance to similarity score (cosine: 0=identical, 2=opposite)
        # similarity = 1 - (distance / 2) gives 0.0 to 1.0 range
        similarity = round(1 - (distance / 2), 3)

        chunks.append({
            "text":        results["documents"][0][i],
            "doc_name":    results["metadatas"][0][i]["doc_name"],
            "doc_id":      results["metadatas"][0][i]["doc_id"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "similarity":  similarity,
        })

    # Sort by similarity descending (most relevant first)
    chunks.sort(key=lambda x: x["similarity"], reverse=True)

    # ── TERMINAL: Print selected chunks in full ────────────────────────────
    query_preview = query[:80] + "..." if len(query) > 80 else query

    print()
    print("=" * 70)
    print("  [RAG] RETRIEVAL RESULT")
    print("  Query : " + query_preview)
    print("  Chunks: " + str(len(chunks)) + " selected from ChromaDB")
    print("=" * 70)

    for rank, c in enumerate(chunks):
        bar_filled = int(c["similarity"] * 20)
        bar        = "█" * bar_filled + "░" * (20 - bar_filled)
        score_pct  = int(c["similarity"] * 100)
        total      = str(c.get("total_chunks", "?"))

        print()
        print("  ┌── SELECTED CHUNK #" + str(rank+1) + " of " + str(len(chunks)) + " ──────────────────────────────────────")
        print("  │  Source      : " + c["doc_name"])
        print("  │  Chunk Index : " + str(c["chunk_index"]) + " of " + total)
        print("  │  Similarity  : [" + bar + "] " + str(score_pct) + "%  (" + str(c["similarity"]) + ")")
        print("  │")
        print("  │  FULL CHUNK CONTENT:")
        print("  │  " + "─" * 58)

        # Print full chunk wrapped at 65 chars per line
        full_text = c["text"].replace("\n", " ").strip()
        words     = full_text.split()
        line      = "  │    "
        for word in words:
            if len(line) + len(word) + 1 > 68:
                print(line)
                line = "  │    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        print("  │  " + "─" * 58)
        print("  │  Chars : " + str(len(c["text"])) + "  |  Words : " + str(len(c["text"].split())))
        print("  └" + "─" * 68)

    print()
    print("  [RAG] " + str(len(chunks)) + " chunks injected into prompt as context for Qwen.")
    print("=" * 70)
    print()

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PROMPT BUILDING
# Combine retrieved chunks + user question into one augmented prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the augmented prompt that gets sent to Qwen.

    Structure:
      "Use the following context to answer the question.
       Only use information from the context provided.

       --- CONTEXT ---
       [Source: story.pdf | Relevance: 0.92]
       The dragon lived in the northern mountains for three centuries...

       [Source: story.pdf | Relevance: 0.87]
       The kingdom of Aldric had been at war with the dragons since...

       --- QUESTION ---
       Write a story about the dragon meeting the king.

       --- ANSWER ---"

    Why this structure works:
      - Model clearly knows what is context vs question
      - Source attribution helps model trust the context
      - "Only use information from context" reduces hallucination
    """
    if not chunks:
        # No relevant chunks found — fall back to normal generation
        return f"User: {query}\nAssistant:"

    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source: {chunk['doc_name']} | Relevance: {chunk['similarity']:.2f}]\n"
            f"{chunk['text']}"
        )

    context_str = "\n\n".join(context_parts)

    prompt = (
        f"Use the following context to answer the question or complete the task. "
        f"Base your response on the provided context.\n\n"
        f"--- CONTEXT ---\n"
        f"{context_str}\n\n"
        f"--- TASK ---\n"
        f"{query}\n\n"
        f"--- RESPONSE ---\n"
    )

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED — Full RAG pipeline in one call
# ══════════════════════════════════════════════════════════════════════════════

def get_rag_context(query: str, top_k: int = TOP_K, doc_id: Optional[str] = None) -> dict:
    """
    Full RAG pipeline: query → retrieve → build prompt

    Returns:
    {
        "augmented_prompt": "Use the following context...",
        "chunks":           [...],          ← retrieved chunks with scores
        "has_context":      True/False,     ← False if no docs indexed yet
        "sources":          ["story.pdf"],  ← which documents were used
    }
    """
    chunks = retrieve_similar(query, top_k=top_k, doc_id=doc_id)
    prompt = build_rag_prompt(query, chunks)
    sources = list({c["doc_name"] for c in chunks})

    return {
        "augmented_prompt": prompt,
        "chunks":           chunks,
        "has_context":      len(chunks) > 0,
        "sources":          sources,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def list_documents() -> list[dict]:
    """
    List all documents currently indexed in ChromaDB.
    Groups chunks by document to show one entry per document.
    """
    if collection is None or collection.count() == 0:
        return []

    # Get all entries to extract unique documents
    all_items = collection.get(include=["metadatas"])
    seen      = {}

    for meta in all_items["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id":       doc_id,
                "doc_name":     meta["doc_name"],
                "total_chunks": meta["total_chunks"],
                "total_chars":  meta.get("char_count", 0) * meta["total_chunks"],
            }

    return list(seen.values())


def delete_document(doc_id: str) -> int:
    """
    Delete all chunks belonging to a specific document from ChromaDB.
    Returns number of chunks deleted.
    """
    if collection is None:
        return 0

    # Get all chunk IDs for this document
    results    = collection.get(where={"doc_id": doc_id})
    chunk_ids  = results["ids"]

    if chunk_ids:
        collection.delete(ids=chunk_ids)
        print(f"[RAG] Deleted {len(chunk_ids)} chunks for doc_id='{doc_id}'")

    return len(chunk_ids)


def get_stats() -> dict:
    """Return RAG stats for the dashboard."""
    if collection is None:
        return {"total_chunks": 0, "total_documents": 0, "ready": False}

    docs = list_documents()
    return {
        "total_chunks":    collection.count(),
        "total_documents": len(docs),
        "documents":       docs,
        "embed_model":     EMBED_MODEL,
        "chunk_size":      CHUNK_SIZE,
        "overlap":         CHUNK_OVERLAP,
        "top_k":           TOP_K,
        "ready":           embedder is not None and collection is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RAG EVALUATION — Check if the answer is correct / grounded
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rag_answer(
    query:    str,
    answer:   str,
    chunks:   list[dict],
) -> dict:
    """
    Evaluate the quality of a RAG-generated answer using 4 metrics.

    METRIC 1 — Context Relevance
      "Were the RIGHT chunks retrieved for this query?"
      Measures: average similarity score of retrieved chunks
      High score = chunks were relevant to the question
      Low score  = wrong chunks were retrieved → answer may be off

    METRIC 2 — Answer Groundedness
      "Is the answer actually based on the retrieved chunks?"
      Measures: how many key words from the answer appear in the chunks
      High score = answer comes from chunks (grounded)
      Low score  = answer comes from model memory (hallucination risk)

    METRIC 3 — Answer Completeness
      "Did the model use the context well?"
      Measures: how many key words from chunks appear in the answer
      High score = model used the context fully
      Low score  = model ignored the context

    METRIC 4 — Query Coverage
      "Does the answer actually address the question asked?"
      Measures: how many key words from query appear in the answer
      High score = answer is on-topic
      Low score  = answer went off-topic

    Overall Score = average of all 4 metrics

    Grade:
      90-100 = Excellent  ✅
      70-89  = Good       ✅
      50-69  = Fair       ⚠️
      Below 50 = Poor     ❌
    """
    import re

    def extract_keywords(text: str) -> set:
        """
        Extract meaningful words — ignore stopwords like 'the', 'is', 'a'.
        These are the words that carry actual meaning.
        """
        stopwords = {
            "the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","to","of","in","on",
            "at","by","for","with","about","against","between","into",
            "through","during","before","after","above","below","from",
            "up","down","out","off","over","under","again","then","once",
            "and","but","or","so","yet","both","either","not","no","nor",
            "just","that","this","these","those","i","you","he","she","it",
            "we","they","what","which","who","whom","his","her","its","my",
            "your","our","their","there","when","where","why","how","all",
        }
        # Find all words of 3+ characters, lowercase
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    # ── METRIC 1: Context Relevance ────────────────────────────────────────
    # Average similarity score of all retrieved chunks
    if chunks:
        avg_similarity    = sum(c["similarity"] for c in chunks) / len(chunks)
        context_relevance = round(avg_similarity * 100, 1)
    else:
        context_relevance = 0.0

    # ── METRIC 2: Answer Groundedness ─────────────────────────────────────
    # What % of key words in the ANSWER also appear in the CHUNKS?
    # High = answer is grounded in chunks (good)
    # Low  = answer uses words not in chunks (possible hallucination)
    answer_keywords = extract_keywords(answer)
    chunk_text_all  = " ".join(c["text"] for c in chunks)
    chunk_keywords  = extract_keywords(chunk_text_all)

    if answer_keywords:
        grounded_words  = answer_keywords & chunk_keywords
        groundedness    = round((len(grounded_words) / len(answer_keywords)) * 100, 1)
    else:
        groundedness    = 0.0

    # ── METRIC 3: Answer Completeness ─────────────────────────────────────
    # What % of key words from CHUNKS appear in the ANSWER?
    # High = model used the context fully
    # Low  = model ignored context despite having it
    if chunk_keywords:
        used_words      = chunk_keywords & answer_keywords
        completeness    = round((len(used_words) / min(len(chunk_keywords), 50)) * 100, 1)
        completeness    = min(completeness, 100.0)
    else:
        completeness    = 0.0

    # ── METRIC 4: Query Coverage ───────────────────────────────────────────
    # What % of key words from the QUERY appear in the ANSWER?
    # High = answer is on-topic
    # Low  = answer went off-topic
    query_keywords  = extract_keywords(query)
    if query_keywords:
        covered_words   = query_keywords & answer_keywords
        query_coverage  = round((len(covered_words) / len(query_keywords)) * 100, 1)
    else:
        query_coverage  = 0.0

    # ── Overall Score ──────────────────────────────────────────────────────
    overall = round(
        (context_relevance * 0.30) +   # 30% weight — were right chunks retrieved?
        (groundedness      * 0.35) +   # 35% weight — is answer from chunks?
        (completeness      * 0.15) +   # 15% weight — did model use chunks well?
        (query_coverage    * 0.20),    # 20% weight — did answer address question?
        1
    )

    # ── Grade ──────────────────────────────────────────────────────────────
    if   overall >= 90: grade, verdict = "A", "✅ Excellent — answer is well grounded in the document"
    elif overall >= 70: grade, verdict = "B", "✅ Good — answer is mostly grounded in the document"
    elif overall >= 50: grade, verdict = "C", "⚠️  Fair — answer partially uses the document context"
    else:               grade, verdict = "D", "❌ Poor — answer may not be based on the document"

    result = {
        "overall_score":      overall,
        "grade":              grade,
        "verdict":            verdict,
        "metrics": {
            "context_relevance":  context_relevance,
            "answer_groundedness": groundedness,
            "answer_completeness": completeness,
            "query_coverage":     query_coverage,
        },
        "details": {
            "chunks_retrieved":     len(chunks),
            "avg_chunk_similarity": round(avg_similarity * 100, 1) if chunks else 0,
            "answer_keywords":      len(answer_keywords),
            "chunk_keywords":       len(chunk_keywords),
            "grounded_words_count": len(grounded_words) if answer_keywords else 0,
        }
    }

    # ── Print evaluation to terminal ───────────────────────────────────────
    print()
    print("=" * 70)
    print("  [RAG EVALUATION]")
    print(f"  Query  : '{query[:70]}{'...' if len(query)>70 else ''}'")
    print(f"  Answer : '{answer[:70]}{'...' if len(answer)>70 else ''}'")
    print("─" * 70)
    print(f"  METRIC 1 — Context Relevance  : {context_relevance:>6.1f}%  (were right chunks retrieved?)")
    print(f"  METRIC 2 — Answer Groundedness: {groundedness:>6.1f}%  (is answer from the chunks?)")
    print(f"  METRIC 3 — Answer Completeness: {completeness:>6.1f}%  (did model use chunks well?)")
    print(f"  METRIC 4 — Query Coverage     : {query_coverage:>6.1f}%  (did answer address question?)")
    print("─" * 70)
    print(f"  OVERALL SCORE : {overall}%  |  GRADE: {grade}")
    print(f"  VERDICT       : {verdict}")
    print("=" * 70)
    print()

    return result