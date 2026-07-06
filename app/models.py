"""
Pydantic request/response models for the Prompt Refiner API.

Defines the data contracts for:
  - POST /onboard  → OnboardRequest / OnboardResponse
  - POST /refine   → RefineRequest  / RefineResponse
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Onboarding
# ──────────────────────────────────────────────────────────────

class OnboardRequest(BaseModel):
    """Payload for creating or updating a user profile."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the user (e.g. 'alice', 'bob_42').",
    )
    domain: Literal["dev", "marketing", "data_analysis"] = Field(
        ...,
        description="The user's working domain. Determines which domain-specific "
                    "best practices are retrieved during prompt refinement.",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Tags describing the user's interests (e.g. ['python', 'rag', 'fastapi']).",
    )
    bio: Optional[str] = Field(
        default=None,
        description="Optional free-text biography. Used to further personalize refined prompts.",
    )


class OnboardResponse(BaseModel):
    """Confirmation returned after successful onboarding."""

    user_id: str
    status: str = "onboarded"


# ──────────────────────────────────────────────────────────────
# Prompt Refinement
# ──────────────────────────────────────────────────────────────

class RefineRequest(BaseModel):
    """Payload for submitting a raw prompt to be refined."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="Must match a previously onboarded user_id.",
    )
    raw_prompt: str = Field(
        ...,
        min_length=1,
        description="The rough, unrefined prompt the user wants improved.",
    )


class RefineResponse(BaseModel):
    """Structured result of the prompt refinement pipeline."""

    refined_prompt: str = Field(
        ...,
        description="The rewritten, improved prompt.",
    )
    explanation: str = Field(
        ...,
        description="1-2 sentence explanation of why the refined prompt is better.",
    )
    technique_used: str = Field(
        ...,
        description="Title of the prompt-engineering technique applied.",
    )
    domain_tip_used: str = Field(
        ...,
        description="Title of the domain-specific best practice applied.",
    )
