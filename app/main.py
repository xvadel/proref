"""
FastAPI application — Personalized Prompt Refiner.

Endpoints:
  GET  /                  → Serves the frontend UI
  POST /onboard           → Creates/updates a user profile
  POST /refine            → Refines a raw prompt using RAG + LLM
  POST /feedback          → Records a thumbs-up/down rating on a refinement
  GET  /history/{user_id} → Returns a user's past refinements

On startup, the knowledge base (prompt techniques + domain tips) is
automatically ingested into ChromaDB, so the app works immediately.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    OnboardRequest, OnboardResponse,
    RefineRequest, RefineResponse,
    FeedbackRequest, FeedbackResponse,
    HistoryResponse,
)
from app.rag import ingest_knowledge_base, store_user_profile, load_user_profile
from app.rag import retrieve_technique, retrieve_domain_tip
from app.llm import refine_prompt
from app.feedback import (
    log_refinement, record_feedback, get_user_history,
    update_chroma_success_rates, maybe_promote_to_kb,
    embed_user_prompt_history, retrieve_similar_user_prompts,
)

# ──────────────────────────────────────────────────────────────
# Load environment variables from .env file
# ──────────────────────────────────────────────────────────────
load_dotenv()


# ──────────────────────────────────────────────────────────────
# App Lifespan — runs on startup/shutdown
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Ingest the knowledge base into ChromaDB, then apply any
             existing feedback-derived success rates to technique metadata.
    Shutdown: (nothing to clean up — ChromaDB persists to disk).
    """
    print("[*] Starting Personalized Prompt Refiner...")
    ingest_knowledge_base()  # also calls update_chroma_success_rates() internally
    print("[+] Ready to refine prompts!")
    yield
    print("[*] Shutting down.")


# ──────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Personalized Prompt Refiner",
    description=(
        "A RAG-powered tool that rewrites vague prompts into specific, effective "
        "prompts — personalized to your domain and interests, with a learning loop "
        "that improves retrieval from user feedback."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static files from the frontend directory
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.get("/", summary="Serve the frontend UI")
async def serve_frontend():
    """Return the single-page frontend application."""
    index_path = _FRONTEND_DIR / "index.html"
    return FileResponse(str(index_path), media_type="text/html")


@app.post("/onboard", response_model=OnboardResponse, summary="Create or update a user profile")
async def onboard_user(request: OnboardRequest):
    """
    Save the user's profile (domain, interests, bio) for personalization.

    The profile is stored both as a JSON file on disk and embedded into
    ChromaDB for potential future semantic retrieval.
    """
    profile = request.model_dump()
    store_user_profile(profile)
    return OnboardResponse(user_id=request.user_id, status="onboarded")


@app.post("/refine", response_model=RefineResponse, summary="Refine a raw prompt")
async def refine_user_prompt(request: RefineRequest):
    """
    Take a rough prompt and rewrite it using RAG-retrieved context + an LLM.

    Pipeline:
      1. Load the user's profile (404 if not onboarded)
      2. Embed and store the raw prompt in the user's prompt history
      3. Retrieve similar past prompts from this user for personalization
      4. Retrieve the best-matching prompt-engineering technique (feedback-aware)
      5. Retrieve the best-matching domain-specific tip
      6. Call the LLM to rewrite the prompt with all context
      7. Log the refinement to the feedback store and return a refinement_id
    """
    # Step 1: Load user profile
    profile = load_user_profile(request.user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.user_id}' not found. Please onboard first via POST /onboard.",
        )

    # Step 2: Store this prompt in the user's history (for future personalization)
    embed_user_prompt_history(
        user_id=request.user_id,
        raw_prompt=request.raw_prompt,
        domain=profile["domain"],
    )

    # Step 3: Retrieve similar past prompts from this user
    similar_prompts = retrieve_similar_user_prompts(
        user_id=request.user_id,
        raw_prompt=request.raw_prompt,
        n=3,
    )

    # Step 4: Retrieve best prompt-engineering technique (feedback-aware reranking)
    technique = retrieve_technique(request.raw_prompt, n_candidates=3)

    # Step 5: Retrieve best domain-specific tip
    domain_tip = retrieve_domain_tip(request.raw_prompt, profile["domain"])

    # Step 6: Call the LLM to refine the prompt
    try:
        refined, explanation = refine_prompt(
            raw_prompt=request.raw_prompt,
            technique=technique,
            domain_tip=domain_tip,
            user_profile=profile,
            similar_past_prompts=similar_prompts,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed: {str(e)}. Check your LLM_PROVIDER and API key settings.",
        )

    technique_title = technique.get("title", "Unknown")
    domain_tip_title = domain_tip.get("title", "Unknown")

    # Step 7: Log the refinement for feedback tracking
    refinement_id = log_refinement(
        user_id=request.user_id,
        raw_prompt=request.raw_prompt,
        refined_prompt=refined,
        technique_used=technique_title,
        domain_tip_used=domain_tip_title,
    )

    return RefineResponse(
        refinement_id=refinement_id,
        refined_prompt=refined,
        explanation=explanation,
        technique_used=technique_title,
        domain_tip_used=domain_tip_title,
    )


@app.post("/feedback", response_model=FeedbackResponse, summary="Rate a refinement result")
async def submit_feedback(request: FeedbackRequest):
    """
    Record a thumbs-up or thumbs-down rating for a past refinement.

    On "good" ratings:
      - Success rates are recomputed and written back to ChromaDB metadata,
        so future retrievals favor techniques that work well.
      - If the refinement meets quality thresholds, it is promoted into the
        knowledge base as a new validated example (self-expanding RAG).

    On "bad" ratings:
      - Success rates are updated to down-weight the technique used.
    """
    found = record_feedback(
        refinement_id=request.refinement_id,
        user_id=request.user_id,
        rating=request.rating,
    )

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Refinement '{request.refinement_id}' not found for user '{request.user_id}'.",
        )

    # Recompute success rates and write back to ChromaDB (lightweight operation)
    update_chroma_success_rates()

    # For "good" ratings: attempt to promote this pair into the KB
    promoted = False
    if request.rating == "good":
        # Fetch the record to get all fields needed for promotion
        history = get_user_history(request.user_id, limit=100)
        record = next(
            (r for r in history if r.get("refinement_id") == request.refinement_id),
            None,
        )
        if record:
            promoted = maybe_promote_to_kb(record)

    return FeedbackResponse(
        refinement_id=request.refinement_id,
        rating=request.rating,
        promoted_to_kb=promoted,
    )


@app.get("/history/{user_id}", response_model=HistoryResponse, summary="Get a user's refinement history")
async def get_history(user_id: str, limit: int = 20):
    """
    Return a user's past refinements (newest first), including ratings.

    Query params:
      limit: max records to return (default 20, max 100).
    """
    # Verify the user exists
    profile = load_user_profile(user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found.",
        )

    limit = min(limit, 100)  # cap to prevent abuse
    entries = get_user_history(user_id, limit=limit)

    return HistoryResponse(
        user_id=user_id,
        total=len(entries),
        entries=entries,
    )
