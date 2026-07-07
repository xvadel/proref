"""
RAG module — ChromaDB setup, knowledge-base ingestion, and retrieval.

This module manages four ChromaDB collections:
  1. prompt_techniques    — general prompt-engineering techniques
                            (with success_rate metadata for feedback-aware reranking)
  2. domain_knowledge     — domain-specific best practices (dev / marketing / data_analysis)
                            + promoted validated examples from real usage
  3. user_profiles        — embedded user profiles
  4. user_prompt_history  — per-user prompt history for personalized retrieval

Embeddings are generated locally using sentence-transformers (all-MiniLM-L6-v2),
so no external API key is needed for the vector search layer.
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Resolve paths relative to the project root (one level up from app/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_PROFILES_DIR = _PROJECT_ROOT / "user_profiles"

# ChromaDB storage path (configurable via .env)
_CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(_PROJECT_ROOT / "chroma_db"))

# Shared embedding function — downloaded automatically on first use (~80 MB)
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Persistent ChromaDB client (data survives server restarts)
chroma_client = chromadb.PersistentClient(path=_CHROMA_DB_PATH)


# ──────────────────────────────────────────────────────────────
# Collection Handles
# ──────────────────────────────────────────────────────────────

def _get_collection(name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection with the shared embedding function."""
    return chroma_client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
    )


# ──────────────────────────────────────────────────────────────
# Knowledge-Base Ingestion
# ──────────────────────────────────────────────────────────────

def ingest_knowledge_base() -> None:
    """
    Load JSON data files and upsert them into ChromaDB collections.

    Called automatically on FastAPI startup so the app works immediately
    without a manual ingestion step. Upserts are idempotent (safe to re-run).
    After ingestion, applies any existing feedback-derived success rates.
    """
    _ingest_prompt_techniques()
    _ingest_domain_knowledge()

    # Apply existing feedback success rates to the freshly ingested techniques
    # (import here to avoid circular import at module level)
    try:
        from app.feedback import update_chroma_success_rates
        update_chroma_success_rates()
    except Exception:
        pass  # No feedback yet — not an error

    print("[+] Knowledge base ingested into ChromaDB.")


def _ingest_prompt_techniques() -> None:
    """Ingest general prompt-engineering techniques."""
    collection = _get_collection("prompt_techniques")
    filepath = _DATA_DIR / "prompt_techniques.json"

    with open(filepath, "r", encoding="utf-8") as f:
        techniques = json.load(f)

    # Build a rich text document for each technique to embed
    ids, documents, metadatas = [], [], []
    for tech in techniques:
        ids.append(tech["id"])
        # Combine title + when_to_use + technique for richer semantic search
        documents.append(
            f"{tech['title']}: {tech['when_to_use']} {tech['technique']}"
        )
        metadatas.append({
            "title": tech["title"],
            "technique": tech["technique"],
            "when_to_use": tech["when_to_use"],
            "example_before": tech["example_before"],
            "example_after": tech["example_after"],
            # Default neutral success rate — overwritten by update_chroma_success_rates()
            # once real feedback data exists.
            "success_rate": 0.5,
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  [i] Ingested {len(ids)} prompt techniques.")


def _ingest_domain_knowledge() -> None:
    """Ingest domain-specific best practices for all domains."""
    collection = _get_collection("domain_knowledge")
    domain_dir = _DATA_DIR / "domain_knowledge"

    ids, documents, metadatas = [], [], []
    for filepath in domain_dir.glob("*.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            ids.append(entry["id"])
            documents.append(f"{entry['title']}: {entry['guidance']}")
            metadatas.append({
                "title": entry["title"],
                "domain": entry["domain"],
                "guidance": entry["guidance"],
            })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  [i] Ingested {len(ids)} domain knowledge entries.")


# ──────────────────────────────────────────────────────────────
# User Profile Storage
# ──────────────────────────────────────────────────────────────

def store_user_profile(profile: dict) -> None:
    """
    Persist a user profile to disk (JSON) and embed it in ChromaDB.

    Args:
        profile: Dict with keys user_id, domain, interests, bio (optional).
    """
    # Ensure the profiles directory exists
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # Save to disk as JSON
    filepath = _PROFILES_DIR / f"{profile['user_id']}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    # Also embed in ChromaDB for potential future semantic retrieval
    collection = _get_collection("user_profiles")
    doc_text = _profile_to_text(profile)
    collection.upsert(
        ids=[profile["user_id"]],
        documents=[doc_text],
        metadatas=[{
            "user_id": profile["user_id"],
            "domain": profile["domain"],
        }],
    )


def load_user_profile(user_id: str) -> dict | None:
    """
    Load a user profile from disk.

    Returns:
        Profile dict if found, None otherwise.
    """
    filepath = _PROFILES_DIR / f"{user_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _profile_to_text(profile: dict) -> str:
    """Convert a profile dict to a searchable text document."""
    parts = [
        f"Domain: {profile['domain']}",
        f"Interests: {', '.join(profile.get('interests', []))}",
    ]
    if profile.get("bio"):
        parts.append(f"Bio: {profile['bio']}")
    return ". ".join(parts)


# ──────────────────────────────────────────────────────────────
# RAG Retrieval — Feedback-Aware Technique Retrieval
# ──────────────────────────────────────────────────────────────

def retrieve_technique(raw_prompt: str, n_candidates: int = 3) -> dict:
    """
    Find the most relevant prompt-engineering technique for the given raw prompt.

    Strategy (feedback-aware reranking):
      1. Retrieve the top `n_candidates` by embedding similarity.
      2. Among candidates whose similarity scores are within a 10% relative
         margin of the best score, pick the one with the highest success_rate.
      3. This means: when two techniques are roughly equally relevant, the one
         that has historically worked better for users wins.

    Returns:
        Dict with keys: title, technique, when_to_use, example_before, example_after,
                        success_rate.
    """
    collection = _get_collection("prompt_techniques")
    # Fetch n_candidates so we have room to rerank
    results = collection.query(
        query_texts=[raw_prompt],
        n_results=min(n_candidates, collection.count() or 1),
        include=["metadatas", "distances"],
    )

    if not results["metadatas"] or not results["metadatas"][0]:
        return _fallback_technique()

    candidates = list(zip(
        results["metadatas"][0],
        results["distances"][0],
    ))

    # Rerank: among candidates within 10% of the best (lowest) distance,
    # prefer the one with the highest success_rate from user feedback.
    best_distance = candidates[0][1]  # ChromaDB returns sorted ascending
    margin = best_distance * 0.10

    close_candidates = [
        (meta, dist) for meta, dist in candidates
        if dist <= best_distance + margin
    ]

    # Sort close candidates by success_rate descending, then by distance ascending
    close_candidates.sort(
        key=lambda x: (-x[0].get("success_rate", 0.5), x[1])
    )

    return close_candidates[0][0]


def retrieve_domain_tip(raw_prompt: str, domain: str) -> dict:
    """
    Find the most relevant domain-specific best practice.

    Filters by domain using ChromaDB's `where` clause to ensure the tip
    matches the user's working domain. Also searches promoted validated
    examples (domain="general") which are domain-agnostic.

    Returns:
        Dict with keys: title, domain, guidance.
    """
    collection = _get_collection("domain_knowledge")

    # Primary: domain-specific tips
    results = collection.query(
        query_texts=[raw_prompt],
        n_results=1,
        where={"domain": domain},
    )

    if results["metadatas"] and results["metadatas"][0]:
        return results["metadatas"][0][0]

    # Fallback: return generic advice
    return {
        "title": "General Best Practice",
        "domain": domain,
        "guidance": "Be specific about your context, goals, and constraints.",
    }


def _fallback_technique() -> dict:
    """Return a safe default technique when the collection is empty."""
    return {
        "title": "Specificity & Constraints",
        "technique": "Add specific details, constraints, and context to your prompt.",
        "when_to_use": "When the prompt is too vague or open-ended.",
        "example_before": "Tell me about X.",
        "example_after": "Explain X in detail, focusing on Y, in under 200 words.",
        "success_rate": 0.5,
    }
