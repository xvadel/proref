"""
Pydantic request/response models for the Prompt Refiner API.

Defines the data contracts for:
  - POST /onboard           → OnboardRequest / OnboardResponse
  - POST /refine            → RefineRequest  / RefineResponse
  - POST /feedback          → FeedbackRequest / FeedbackResponse
  - POST /translate-refine  → TranslateRefineRequest / TranslateRefineResponse
  - GET  /history/{id}      → HistoryResponse

Profile fields (v2.1):
  Core    — user_id, domain (6 options), interests, bio
  New     — experience_level, preferred_language, tone_preference,
            goal, tools, output_format_preference, avoid_topics
"""

import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────
# Onboarding
# ──────────────────────────────────────────────────────────────

class OnboardRequest(BaseModel):
    """Payload for creating or updating a user profile.

    Core fields:
        user_id           — unique identifier
        domain            — working domain (6 options)
        interests         — list of interest/skill tags
        bio               — optional free-text bio

    Extended personalization fields (v2.1):
        experience_level       — calibrates prompt complexity and vocabulary
        preferred_language     — language the refined prompt should be written in
        tone_preference        — style register of the refined output
        goal                   — the user's overarching objective (free-text)
        tools                  — specific tools/technologies the user works with
        output_format_preference — preferred structure of the refined prompt's output
        avoid_topics           — topics or patterns the user wants excluded
    """

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the user (e.g. 'alice', 'bob_42').",
    )
    domain: Literal[
        "dev",
        "marketing",
        "data_analysis",
        "education",
        "research",
        "creative_writing",
    ] = Field(
        ...,
        description=(
            "The user's working domain. Determines which domain-specific "
            "best practices are retrieved during prompt refinement. "
            "Supported: dev, marketing, data_analysis, education, research, creative_writing."
        ),
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Tags describing the user's interests (e.g. ['python', 'rag', 'fastapi']).",
    )
    bio: Optional[str] = Field(
        default=None,
        description="Optional free-text biography. Used to further personalize refined prompts.",
    )

    # ── Extended personalization fields (v2.1) ─────────────────────────────

    experience_level: Literal["beginner", "intermediate", "expert"] = Field(
        default="intermediate",
        description=(
            "The user's expertise level. Controls vocabulary complexity and assumed "
            "prior knowledge in the refined prompt. "
            "'beginner' → simple language, more explanation; "
            "'expert' → technical terminology, concise."
        ),
    )
    preferred_language: str = Field(
        default="English",
        description=(
            "The language the refined prompt (and ideally the LLM response) should be "
            "written in. E.g. 'English', 'Arabic', 'French', 'Spanish'."
        ),
    )
    tone_preference: Literal["formal", "balanced", "casual"] = Field(
        default="balanced",
        description=(
            "Preferred style register for the refined prompt. "
            "'formal' → professional/academic; "
            "'balanced' → clear and approachable (default); "
            "'casual' → conversational and direct."
        ),
    )
    goal: Optional[str] = Field(
        default=None,
        description=(
            "The user's overarching objective or project context. Free-text. "
            "E.g. 'Building a RAG-based chatbot for my final-year project'. "
            "Helps the LLM tailor refinements toward a broader purpose."
        ),
    )
    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Specific tools, libraries, or technologies the user works with. "
            "E.g. ['FastAPI', 'PostgreSQL', 'Docker']. "
            "Injected into the LLM context so refinements reference relevant tooling."
        ),
    )
    output_format_preference: Literal[
        "paragraph", "bullets", "structured", "code"
    ] = Field(
        default="paragraph",
        description=(
            "Preferred structure for the output requested in the refined prompt. "
            "'paragraph' → flowing prose; 'bullets' → concise bullet list; "
            "'structured' → headings/sections; 'code' → code-first with minimal prose."
        ),
    )
    avoid_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Topics, patterns, or styles the user wants excluded from refinements. "
            "E.g. ['jargon', 'passive voice', 'generic advice']. "
            "Added as negative constraints to the LLM prompt."
        ),
    )

    @field_validator("user_id")
    @classmethod
    def sanitize_user_id(cls, v: str) -> str:
        """
        Strip everything except alphanumerics, underscores, and hyphens.
        Prevents path-traversal attacks when user_id is used to build file paths.
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", v).strip("_")
        if not sanitized:
            raise ValueError("user_id contains no valid characters.")
        return sanitized


class OnboardResponse(BaseModel):
    """Confirmation returned after successful onboarding."""

    user_id: str
    domain: str
    experience_level: str
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
        min_length=5,
        description="The rough, unrefined prompt the user wants improved. Must be at least 5 characters.",
    )

    @field_validator("user_id")
    @classmethod
    def sanitize_user_id(cls, v: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", v).strip("_")
        if not sanitized:
            raise ValueError("user_id contains no valid characters.")
        return sanitized


class RefineResponse(BaseModel):
    """Structured result of the prompt refinement pipeline."""

    refinement_id: str = Field(
        ...,
        description="UUID for this refinement. Send it back with POST /feedback to rate the result.",
    )
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


# ──────────────────────────────────────────────────────────────
# Feedback
# ──────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Payload for rating a refinement result."""

    refinement_id: str = Field(
        ...,
        description="UUID returned by POST /refine or POST /translate-refine.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="Must match the user who made the refinement.",
    )
    rating: Literal["good", "bad"] = Field(
        ...,
        description="'good' = used as-is or improved it. 'bad' = not useful.",
    )

    @field_validator("user_id")
    @classmethod
    def sanitize_user_id(cls, v: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", v).strip("_")
        if not sanitized:
            raise ValueError("user_id contains no valid characters.")
        return sanitized


class FeedbackResponse(BaseModel):
    """Confirmation returned after recording feedback."""

    refinement_id: str
    rating: str
    promoted_to_kb: bool = Field(
        default=False,
        description="True if this 'good' refinement was promoted into the knowledge base.",
    )


# ──────────────────────────────────────────────────────────────
# Arabic Translate + Refine
# ──────────────────────────────────────────────────────────────

class TranslateRefineRequest(BaseModel):
    """Payload for translating an Arabic prompt and refining it into professional English."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="Must match a previously onboarded user_id.",
    )
    arabic_prompt: str = Field(
        ...,
        min_length=5,
        description="The raw Arabic prompt to translate and refine.",
    )

    @field_validator("user_id")
    @classmethod
    def sanitize_user_id(cls, v: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", v).strip("_")
        if not sanitized:
            raise ValueError("user_id contains no valid characters.")
        return sanitized


class TranslateRefineResponse(BaseModel):
    """Structured result of the Arabic translate-and-refine pipeline."""

    refinement_id: str = Field(
        ...,
        description="UUID for this refinement — can be rated via POST /feedback.",
    )
    original_arabic: str = Field(
        ...,
        description="The original Arabic prompt as submitted.",
    )
    translated_english: str = Field(
        ...,
        description="Literal English translation of the Arabic prompt.",
    )
    refined_prompt: str = Field(
        ...,
        description="The fully refined, professional English prompt.",
    )
    explanation: str = Field(
        ...,
        description="Why the refined prompt is better than the raw translation.",
    )
    technique_used: str
    domain_tip_used: str


# ──────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    """A single past refinement in a user's history."""

    refinement_id: str
    raw_prompt: str
    refined_prompt: str
    technique_used: str
    domain_tip_used: str
    rating: Optional[Literal["good", "bad"]] = None
    timestamp: str
    source: Optional[str] = Field(
        default="refine",
        description="Which pipeline produced this entry: 'refine' or 'translate-refine'.",
    )


class HistoryResponse(BaseModel):
    """A user's past refinement history."""

    user_id: str
    total: int
    entries: list[HistoryEntry]


# ──────────────────────────────────────────────────────────────
# Prompt Chat (iterative refinement)
# ──────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in a prompt-chat conversation."""

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Who sent this message: 'user' or 'assistant'.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The message content.",
    )


class ChatRequest(BaseModel):
    """Payload for sending a follow-up message in a refinement chat session."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="Must match a previously onboarded user_id.",
    )
    refined_prompt: str = Field(
        ...,
        min_length=5,
        description="The current refined prompt being iterated on.",
    )
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Full conversation history so far (user + assistant turns).",
    )
    new_message: str = Field(
        ...,
        min_length=1,
        description="The user's latest instruction or question about the refined prompt.",
    )

    @field_validator("user_id")
    @classmethod
    def sanitize_user_id(cls, v: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", v).strip("_")
        if not sanitized:
            raise ValueError("user_id contains no valid characters.")
        return sanitized


class ChatResponse(BaseModel):
    """The assistant's reply in a prompt-chat session."""

    reply: str = Field(
        ...,
        description="The assistant's response or updated prompt version.",
    )
    updated_prompt: Optional[str] = Field(
        default=None,
        description="If the assistant produced an updated version of the prompt, it appears here.",
    )
