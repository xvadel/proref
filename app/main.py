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
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    OnboardRequest, OnboardResponse,
    RefineRequest, RefineResponse,
    FeedbackRequest, FeedbackResponse,
    HistoryResponse,
    TranslateRefineRequest, TranslateRefineResponse,
)
from app.rag import ingest_knowledge_base, store_user_profile, load_user_profile
from app.rag import retrieve_technique, retrieve_domain_tip
from app.llm import refine_prompt, translate_prompt_to_english, translate_and_refine
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
    return OnboardResponse(
        user_id=request.user_id,
        domain=request.domain,
        experience_level=request.experience_level,
        status="onboarded",
    )


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    """Favicon route to avoid 404 logs in browsers."""
    favicon_path = _FRONTEND_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    # Return 204 No Content if no file exists to keep logs clean
    from fastapi.responses import Response
    return Response(status_code=204)


@app.post("/refine", response_model=RefineResponse, summary="Refine a raw prompt")
async def refine_user_prompt(request: RefineRequest):
    """
    Take a rough prompt and rewrite it using RAG-retrieved context + an LLM.

    Pipeline:
      1. Load the user's profile (404 if not onboarded)
      2. Retrieve similar past prompts from this user for personalization
      3. Embed and store the raw prompt in the user's prompt history (after retrieving)
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

    # Step 2: Retrieve similar past prompts from this user (retrieve BEFORE embedding to avoid self-match)
    similar_prompts = retrieve_similar_user_prompts(
        user_id=request.user_id,
        raw_prompt=request.raw_prompt,
        n=3,
    )

    # Step 3: Store this prompt in the user's history
    embed_user_prompt_history(
        user_id=request.user_id,
        raw_prompt=request.raw_prompt,
        domain=profile["domain"],
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

    # Step 7: Log the refinement for feedback tracking (pass domain for KB promotion)
    refinement_id = log_refinement(
        user_id=request.user_id,
        raw_prompt=request.raw_prompt,
        refined_prompt=refined,
        technique_used=technique_title,
        domain_tip_used=domain_tip_title,
        domain=profile["domain"],
        source="refine",
    )

    return RefineResponse(
        refinement_id=refinement_id,
        refined_prompt=refined,
        explanation=explanation,
        technique_used=technique_title,
        domain_tip_used=domain_tip_title,
    )


@app.post("/translate-refine", response_model=TranslateRefineResponse, summary="Translate Arabic prompt and refine it in English")
async def translate_and_refine_arabic_prompt(request: TranslateRefineRequest):
    """
    Two-step composition pipeline for Arabic prompts:
      1. Translate the raw Arabic prompt into literal English.
      2. Run standard RAG on the translated English text to retrieve technique & domain tips.
      3. Retrieve similar past English prompts from the user.
      4. Embed the translated English text into the user's prompt history.
      5. Call the LLM to rewrite the translated English prompt into a polished professional prompt.
      6. Log the refinement as source="translate-refine".
    """
    # Step 1: Load user profile
    profile = load_user_profile(request.user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.user_id}' not found. Please onboard first via POST /onboard.",
        )

    # Step 2: Translate Arabic → English
    try:
        translated_english = translate_prompt_to_english(request.arabic_prompt)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM translation failed: {str(e)}",
        )

    # Step 3: Retrieve similar past prompts (using the English translation for cross-match)
    similar_prompts = retrieve_similar_user_prompts(
        user_id=request.user_id,
        raw_prompt=translated_english,
        n=3,
    )

    # Step 4: Store the English translation in the user's prompt history
    embed_user_prompt_history(
        user_id=request.user_id,
        raw_prompt=translated_english,
        domain=profile["domain"],
    )

    # Step 5: Retrieve best prompt technique & domain tip (using the English translation!)
    technique = retrieve_technique(translated_english, n_candidates=3)
    domain_tip = retrieve_domain_tip(translated_english, profile["domain"])

    # Step 6: Perform refinement referencing both original Arabic and translated English
    try:
        refined, explanation = translate_and_refine(
            arabic_prompt=request.arabic_prompt,
            translated_english=translated_english,
            technique=technique,
            domain_tip=domain_tip,
            user_profile=profile,
            similar_past_prompts=similar_prompts,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM refinement failed: {str(e)}",
        )

    technique_title = technique.get("title", "Unknown")
    domain_tip_title = domain_tip.get("title", "Unknown")

    # Step 7: Log refinement
    refinement_id = log_refinement(
        user_id=request.user_id,
        raw_prompt=translated_english,  # log English version as base
        refined_prompt=refined,
        technique_used=technique_title,
        domain_tip_used=domain_tip_title,
        domain=profile["domain"],
        source="translate-refine",
    )

    return TranslateRefineResponse(
        refinement_id=refinement_id,
        original_arabic=request.arabic_prompt,
        translated_english=translated_english,
        refined_prompt=refined,
        explanation=explanation,
        technique_used=technique_title,
        domain_tip_used=domain_tip_title,
    )


@app.post("/feedback", response_model=FeedbackResponse, summary="Rate a refinement result")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Record a thumbs-up or thumbs-down rating for a past refinement.

    On "good" ratings:
      - Success rates are recomputed in the background and written back to ChromaDB metadata.
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

    # FIX: Recompute success rates in a background thread to prevent blocking the response
    background_tasks.add_task(update_chroma_success_rates)

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
