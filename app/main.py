"""
FastAPI application — Personalized Prompt Refiner.

Endpoints:
  GET  /          → Serves the frontend UI
  POST /onboard   → Creates/updates a user profile
  POST /refine    → Refines a raw prompt using RAG + LLM

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

from app.models import OnboardRequest, OnboardResponse, RefineRequest, RefineResponse
from app.rag import ingest_knowledge_base, store_user_profile, load_user_profile
from app.rag import retrieve_technique, retrieve_domain_tip
from app.llm import refine_prompt

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
    Startup: Ingest the knowledge base into ChromaDB.
    Shutdown: (nothing to clean up — ChromaDB persists to disk).
    """
    print("🚀 Starting Personalized Prompt Refiner...")
    ingest_knowledge_base()
    print("✅ Ready to refine prompts!")
    yield
    print("👋 Shutting down.")


# ──────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Personalized Prompt Refiner",
    description=(
        "A RAG-powered tool that rewrites vague prompts into specific, effective "
        "prompts — personalized to your domain and interests."
    ),
    version="1.0.0",
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
      2. Retrieve the best-matching prompt-engineering technique
      3. Retrieve the best-matching domain-specific tip
      4. Call the LLM to rewrite the prompt
      5. Return the refined prompt + explanation + metadata
    """
    # Step 1: Load user profile
    profile = load_user_profile(request.user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.user_id}' not found. Please onboard first via POST /onboard.",
        )

    # Step 2: Retrieve the best prompt-engineering technique
    technique = retrieve_technique(request.raw_prompt)

    # Step 3: Retrieve the best domain-specific tip
    domain_tip = retrieve_domain_tip(request.raw_prompt, profile["domain"])

    # Step 4: Call the LLM to refine the prompt
    try:
        refined, explanation = refine_prompt(
            raw_prompt=request.raw_prompt,
            technique=technique,
            domain_tip=domain_tip,
            user_profile=profile,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed: {str(e)}. Check your LLM_PROVIDER and API key settings.",
        )

    # Step 5: Return structured response
    return RefineResponse(
        refined_prompt=refined,
        explanation=explanation,
        technique_used=technique.get("title", "Unknown"),
        domain_tip_used=domain_tip.get("title", "Unknown"),
    )
