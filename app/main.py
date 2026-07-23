"""
FastAPI application — Personalized Prompt Refiner.

Endpoints:
  GET  /                  → Serves the frontend UI
  POST /onboard           → Creates/updates a user profile
  POST /refine            → Refines a raw prompt using RAG + LLM + HF domain signals
  POST /translate-refine  → Translates an Arabic prompt (HF/LLM) then refines
  POST /feedback          → Records a thumbs-up/down rating on a refinement
  GET  /history/{user_id} → Returns a user's past refinements

On startup:
  - The knowledge base (prompt techniques + domain tips) is automatically
    ingested into ChromaDB.
  - Feedback-derived success rates are reapplied to technique metadata.
  - The logging infrastructure is initialised.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Load .env before importing anything that reads settings
load_dotenv()

from app.logging_config import get_logger
from app.models import (
    FeedbackRequest, FeedbackResponse,
    HistoryResponse,
    OnboardRequest, OnboardResponse,
    RefineRequest, RefineResponse,
    TranslateRefineRequest, TranslateRefineResponse,
)
from app.rag import ingest_knowledge_base, load_user_profile, store_user_profile
from app.rag import retrieve_domain_tip, retrieve_technique
from app.llm import refine_prompt, translate_and_refine, translate_prompt_to_english
from app.feedback import (
    embed_user_prompt_history, get_user_history, log_refinement,
    maybe_promote_to_kb, record_feedback, retrieve_similar_user_prompts,
    update_chroma_success_rates,
)

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# App Lifespan — runs on startup/shutdown
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Ingest the knowledge base into ChromaDB, then apply any
             existing feedback-derived success rates to technique metadata.
    Shutdown: Log the shutdown event (ChromaDB persists to disk automatically).
    """
    logger.info("Starting Personalized Prompt Refiner...")
    ingest_knowledge_base()
    logger.info("Server ready — all systems operational.")
    yield
    logger.info("Server shutting down gracefully.")


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
    version="3.0.0",
    lifespan=lifespan,
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ──────────────────────────────────────────────────────────────
# Routes — Static Assets
# ──────────────────────────────────────────────────────────────

@app.get("/", summary="Serve the frontend UI")
async def serve_frontend():
    """Return the single-page frontend application."""
    return FileResponse(str(_FRONTEND_DIR / "index.html"), media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    """Favicon route to avoid 404 logs in browsers."""
    favicon_path = _FRONTEND_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    from fastapi.responses import Response
    return Response(status_code=204)


# ──────────────────────────────────────────────────────────────
# Routes — Onboarding
# ──────────────────────────────────────────────────────────────

@app.post("/onboard", response_model=OnboardResponse, summary="Create or update a user profile")
async def onboard_user(request: OnboardRequest):
    """
    Save the user's profile (domain, interests, bio, and all v2.1 fields)
    for personalisation. Profile is stored on disk and embedded into ChromaDB.
    """
    logger.info("Onboarding user | user_id=%s | domain=%s", request.user_id, request.domain)
    profile = request.model_dump()
    store_user_profile(profile)
    return OnboardResponse(
        user_id=request.user_id,
        domain=request.domain,
        experience_level=request.experience_level,
        status="onboarded",
    )


# ──────────────────────────────────────────────────────────────
# Routes — Prompt Refinement
# ──────────────────────────────────────────────────────────────

@app.post("/refine", response_model=RefineResponse, summary="Refine a raw prompt")
async def refine_user_prompt(request: RefineRequest):
    """
    Take a rough prompt and rewrite it using RAG + HuggingFace domain signals + LLM.

    Pipeline:
      1. Load the user's profile (404 if not onboarded)
      2. Retrieve similar past prompts for personalisation
      3. Embed the raw prompt into the user's prompt history
      4. Retrieve the best prompt-engineering technique (feedback-aware)
      5. Retrieve the best domain-specific tip
      6. Extract HuggingFace specialist signals for the user's domain
      7. Call the LLM with retry/timeout to produce the refined prompt
      8. Log the refinement and return a refinement_id
    """
    logger.info(
        "POST /refine | user_id=%s | prompt_chars=%d",
        request.user_id,
        len(request.raw_prompt),
    )

    # Step 1: Load user profile
    profile = load_user_profile(request.user_id)
    if profile is None:
        logger.warning("Refine attempted for unknown user | user_id=%s", request.user_id)
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.user_id}' not found. Please onboard first via POST /onboard.",
        )

    # Step 2: Retrieve similar past prompts (retrieve BEFORE embedding to avoid self-match)
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

    # Step 6–7: Call the LLM (HF signals extracted internally in refine_prompt)
    try:
        refined, explanation = refine_prompt(
            raw_prompt=request.raw_prompt,
            technique=technique,
            domain_tip=domain_tip,
            user_profile=profile,
            similar_past_prompts=similar_prompts,
        )
    except Exception as exc:
        logger.error(
            "LLM refinement failed | user=%s | error=%s", request.user_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed: {exc}. Check your LLM_PROVIDER and API key settings.",
        )

    technique_title = technique.get("title", "Unknown")
    domain_tip_title = domain_tip.get("title", "Unknown")

    # Step 8: Log the refinement
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


@app.post(
    "/translate-refine",
    response_model=TranslateRefineResponse,
    summary="Translate Arabic prompt and refine it in English",
)
async def translate_and_refine_arabic_prompt(request: TranslateRefineRequest):
    """
    Two-step composition pipeline for Arabic prompts:
      1. Translate Arabic → English (HuggingFace Helsinki-NLP primary, LLM fallback).
      2. Run RAG on the English translation to retrieve technique & domain tip.
      3. Retrieve similar past English prompts from the user.
      4. Embed the translated English text into the user's prompt history.
      5. Call the LLM to rewrite the translation into a polished professional prompt.
      6. Log the refinement as source="translate-refine".
    """
    logger.info(
        "POST /translate-refine | user_id=%s | arabic_chars=%d",
        request.user_id,
        len(request.arabic_prompt),
    )

    # Step 1: Load user profile
    profile = load_user_profile(request.user_id)
    if profile is None:
        logger.warning("Translate-refine attempted for unknown user | user_id=%s", request.user_id)
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.user_id}' not found. Please onboard first via POST /onboard.",
        )

    # Step 2: Translate Arabic → English (HF primary, LLM fallback)
    try:
        translated_english = translate_prompt_to_english(request.arabic_prompt)
    except Exception as exc:
        logger.error("Translation failed | user=%s | error=%s", request.user_id, exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Translation failed: {exc}",
        )

    logger.info(
        "Translation complete | english_chars=%d", len(translated_english)
    )

    # Step 3: Retrieve similar past prompts (using English translation for cross-match)
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

    # Step 5: Retrieve technique & domain tip (using English translation)
    technique = retrieve_technique(translated_english, n_candidates=3)
    domain_tip = retrieve_domain_tip(translated_english, profile["domain"])

    # Step 6: Perform refinement (HF signals extracted internally)
    try:
        refined, explanation = translate_and_refine(
            arabic_prompt=request.arabic_prompt,
            translated_english=translated_english,
            technique=technique,
            domain_tip=domain_tip,
            user_profile=profile,
            similar_past_prompts=similar_prompts,
        )
    except Exception as exc:
        logger.error(
            "Arabic refinement failed | user=%s | error=%s", request.user_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=502,
            detail=f"LLM refinement failed: {exc}",
        )

    technique_title = technique.get("title", "Unknown")
    domain_tip_title = domain_tip.get("title", "Unknown")

    # Step 7: Log refinement
    refinement_id = log_refinement(
        user_id=request.user_id,
        raw_prompt=translated_english,
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


# ──────────────────────────────────────────────────────────────
# Routes — Feedback
# ──────────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse, summary="Rate a refinement result")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Record a thumbs-up or thumbs-down rating for a past refinement.

    On "good" ratings:
      - Success rates are recomputed in the background and written back to ChromaDB.
      - If the refinement meets quality thresholds, it is promoted into the
        knowledge base as a new validated example (self-expanding RAG).

    On "bad" ratings:
      - Success rates are updated to down-weight the technique used.
    """
    logger.info(
        "POST /feedback | refinement_id=%s | user=%s | rating=%s",
        request.refinement_id[:8],
        request.user_id,
        request.rating,
    )

    found = record_feedback(
        refinement_id=request.refinement_id,
        user_id=request.user_id,
        rating=request.rating,
    )

    if not found:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Refinement '{request.refinement_id}' not found for user "
                f"'{request.user_id}'."
            ),
        )

    # Recompute success rates in a background thread (non-blocking)
    background_tasks.add_task(update_chroma_success_rates)

    promoted = False
    if request.rating == "good":
        history = get_user_history(request.user_id, limit=100)
        record = next(
            (r for r in history if r.get("refinement_id") == request.refinement_id),
            None,
        )
        if record:
            promoted = maybe_promote_to_kb(record)
            if promoted:
                logger.info(
                    "Refinement promoted to KB | refinement_id=%s", request.refinement_id[:8]
                )

    return FeedbackResponse(
        refinement_id=request.refinement_id,
        rating=request.rating,
        promoted_to_kb=promoted,
    )


# ──────────────────────────────────────────────────────────────
# Routes — History
# ──────────────────────────────────────────────────────────────

@app.get(
    "/history/{user_id}",
    response_model=HistoryResponse,
    summary="Get a user's refinement history",
)
async def get_history(user_id: str, limit: int = 20):
    """
    Return a user's past refinements (newest first), including ratings.

    Query params:
      limit: max records to return (default 20, max 100).
    """
    profile = load_user_profile(user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found.",
        )

    limit = min(limit, 100)
    entries = get_user_history(user_id, limit=limit)
    logger.debug("GET /history/%s | returned=%d", user_id, len(entries))

    return HistoryResponse(
        user_id=user_id,
        total=len(entries),
        entries=entries,
    )
