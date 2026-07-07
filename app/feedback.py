"""
Feedback module — stores refinement feedback and computes technique success rates.

This module is the foundation of the learning loop:
  1. Every /refine response gets a unique refinement_id.
  2. The user can POST /feedback with a rating ("good" | "bad").
  3. /history/{user_id} returns a user's past refinements.
  4. compute_technique_success_rates() aggregates good/total ratios per technique
     and writes them back to the prompt_techniques ChromaDB collection as metadata,
     so retrieval can use success_rate as a tiebreaker when similarity scores are close.

Storage: one JSONL file per user at feedback/{user_id}.jsonl  (append-only log,
simple and durable — no database needed for a portfolio project).
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FEEDBACK_DIR = _PROJECT_ROOT / "feedback"
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Refinement Log — write/read
# ──────────────────────────────────────────────────────────────

def log_refinement(
    user_id: str,
    raw_prompt: str,
    refined_prompt: str,
    technique_used: str,
    domain_tip_used: str,
) -> str:
    """
    Append a new refinement entry to the user's feedback log.

    Returns:
        refinement_id: a UUID that the client sends back with the rating.
    """
    refinement_id = str(uuid.uuid4())
    record = {
        "refinement_id": refinement_id,
        "user_id": user_id,
        "raw_prompt": raw_prompt,
        "refined_prompt": refined_prompt,
        "technique_used": technique_used,
        "domain_tip_used": domain_tip_used,
        "rating": None,          # populated later via POST /feedback
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _append_record(user_id, record)
    return refinement_id


def record_feedback(refinement_id: str, user_id: str, rating: str) -> bool:
    """
    Update the rating field of an existing refinement record.

    Args:
        refinement_id: UUID returned by /refine.
        user_id:       Must match the user who made the refinement.
        rating:        "good" or "bad".

    Returns:
        True if the record was found and updated, False if not found.
    """
    log_path = _log_path(user_id)
    if not log_path.exists():
        return False

    records = _read_records(user_id)
    updated = False
    for rec in records:
        if rec.get("refinement_id") == refinement_id:
            rec["rating"] = rating
            updated = True
            break

    if updated:
        # Rewrite the file with the updated record
        with open(log_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return updated


def get_user_history(user_id: str, limit: int = 20) -> list[dict]:
    """
    Return the most recent refinements for a user (newest first).

    Args:
        user_id: The user whose history to fetch.
        limit:   Maximum number of records to return.
    """
    records = _read_records(user_id)
    # Reverse so newest is first, then cap
    return list(reversed(records))[:limit]


# ──────────────────────────────────────────────────────────────
# Technique Success-Rate Computation
# ──────────────────────────────────────────────────────────────

def compute_technique_success_rates() -> dict[str, float]:
    """
    Scan all feedback logs and compute a success_rate per technique title.

    success_rate = good_ratings / total_rated_uses  (0.0–1.0)
    Techniques with no rated uses get 0.5 (neutral prior).

    Returns:
        Dict mapping technique_title -> success_rate float.
    """
    good_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)

    # Walk every user's feedback file
    for log_file in _FEEDBACK_DIR.glob("*.jsonl"):
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            technique = rec.get("technique_used", "")
            rating = rec.get("rating")

            if not technique or rating is None:
                continue  # skip unrated

            total_counts[technique] += 1
            if rating == "good":
                good_counts[technique] += 1

    rates: dict[str, float] = {}
    for technique, total in total_counts.items():
        rates[technique] = round(good_counts[technique] / total, 3)

    return rates


def update_chroma_success_rates() -> None:
    """
    Recompute success rates and write them back to the prompt_techniques
    ChromaDB collection as metadata. Called on startup and after each feedback.

    This is what makes retrieval feedback-aware: retrieve_technique() reads
    success_rate from metadata and uses it to rerank close candidates.
    """
    # Import here to avoid circular import (rag imports feedback, feedback imports rag)
    from app.rag import _get_collection

    rates = compute_technique_success_rates()
    if not rates:
        return  # No rated data yet — nothing to update

    collection = _get_collection("prompt_techniques")
    existing = collection.get(include=["metadatas", "documents"])

    if not existing["ids"]:
        return

    # Rebuild with updated success_rate metadata
    ids, documents, metadatas = [], [], []
    for i, doc_id in enumerate(existing["ids"]):
        meta = existing["metadatas"][i].copy()
        title = meta.get("title", "")
        # Update success_rate if we have data for this technique; default 0.5
        meta["success_rate"] = rates.get(title, 0.5)
        ids.append(doc_id)
        documents.append(existing["documents"][i])
        metadatas.append(meta)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[+] Updated success_rate for {len(rates)} techniques in ChromaDB.")


# ──────────────────────────────────────────────────────────────
# Knowledge-Base Promotion (self-expanding RAG)
# ──────────────────────────────────────────────────────────────

def maybe_promote_to_kb(record: dict) -> bool:
    """
    If a refinement gets a "good" rating, consider promoting the
    (raw_prompt -> refined_prompt) pair into the KB as a new few-shot example.

    Promotion criteria (conservative — avoids polluting the KB with noise):
      - rating == "good"
      - raw_prompt is at least 10 chars (not a trivial test)
      - refined_prompt is at least 30 chars
      - The pair hasn't already been promoted (idempotent)

    Promoted examples are appended to data/promoted_examples.jsonl and
    upserted into the domain_knowledge collection as domain-tagged entries.

    Returns:
        True if the record was promoted, False otherwise.
    """
    from app.rag import _get_collection

    if record.get("rating") != "good":
        return False

    raw = record.get("raw_prompt", "")
    refined = record.get("refined_prompt", "")
    if len(raw) < 10 or len(refined) < 30:
        return False

    # Idempotency: use refinement_id as Chroma doc ID with "promoted_" prefix
    doc_id = f"promoted_{record['refinement_id']}"
    collection = _get_collection("domain_knowledge")

    # Check if already promoted
    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        return False  # already in KB

    # Build a natural-language example entry
    domain = record.get("domain_tip_used", "general")
    # domain_tip_used is a title like "Specify Tech Stack & Constraints" —
    # we'll tag it under the user's domain by reading the profile
    # (for simplicity, we store it as a "validated_example" domain entry)
    doc_text = (
        f"Validated example — Before: {raw} | "
        f"After: {refined} | "
        f"Technique: {record.get('technique_used', '')} | "
        f"Tip: {record.get('domain_tip_used', '')}"
    )
    meta = {
        "title": f"Validated example: {raw[:60]}...",
        "domain": "general",  # searchable across all domains
        "guidance": f"Technique: {record.get('technique_used','')}. "
                    f"Good refinement of: '{raw[:80]}' → '{refined[:80]}'",
        "is_promoted": True,
        "source_refinement_id": record["refinement_id"],
    }

    collection.upsert(ids=[doc_id], documents=[doc_text], metadatas=[meta])

    # Also persist to disk for auditability
    promoted_log = _PROJECT_ROOT / "data" / "promoted_examples.jsonl"
    with open(promoted_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({**record, "chroma_id": doc_id}, ensure_ascii=False) + "\n")

    print(f"[+] Promoted refinement {record['refinement_id'][:8]}... to knowledge base.")
    return True


# ──────────────────────────────────────────────────────────────
# User Prompt History (for adaptive personalization)
# ──────────────────────────────────────────────────────────────

def embed_user_prompt_history(user_id: str, raw_prompt: str, domain: str) -> None:
    """
    Embed a user's raw prompt into the user_prompt_history ChromaDB collection.

    This enables retrieve_similar_user_prompts() to find patterns in a user's
    past prompts, which gets injected as context to the LLM for deeper personalization.
    """
    from app.rag import _get_collection

    collection = _get_collection("user_prompt_history")
    # Use a timestamp-based ID so we store multiple prompts per user
    doc_id = f"{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    collection.upsert(
        ids=[doc_id],
        documents=[raw_prompt],
        metadatas=[{"user_id": user_id, "domain": domain}],
    )


def retrieve_similar_user_prompts(user_id: str, raw_prompt: str, n: int = 3) -> list[str]:
    """
    Find the user's past prompts most similar to the current one.

    Returns a list of raw prompt strings (empty list if no history yet).
    """
    from app.rag import _get_collection

    collection = _get_collection("user_prompt_history")

    # Check if user has any history
    existing = collection.get(where={"user_id": user_id})
    if not existing["ids"]:
        return []

    results = collection.query(
        query_texts=[raw_prompt],
        n_results=min(n, len(existing["ids"])),
        where={"user_id": user_id},
    )

    if results["documents"] and results["documents"][0]:
        # Filter out the current prompt if it's an exact match
        return [d for d in results["documents"][0] if d != raw_prompt]

    return []


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _log_path(user_id: str) -> Path:
    """Return the JSONL log file path for a user."""
    return _FEEDBACK_DIR / f"{user_id}.jsonl"


def _append_record(user_id: str, record: dict) -> None:
    """Append a JSON record to the user's log file (one record per line)."""
    with open(_log_path(user_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_records(user_id: str) -> list[dict]:
    """Read all records from a user's log file."""
    log_path = _log_path(user_id)
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records
