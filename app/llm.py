"""
LLM client module — pluggable adapter for Groq and Ollama backends.

Provider is selected via the LLM_PROVIDER environment variable:
  - "groq"   → Uses the Groq Python SDK (free tier, needs GROQ_API_KEY)
  - "ollama" → Calls Ollama's local REST API (no API key, needs Ollama running)

The main entry point is `refine_prompt()`, which:
  1. Builds a system prompt instructing the LLM to rewrite the user's raw prompt
  2. Injects the user's similar past prompts for deeper personalization
  3. Sends it to the configured LLM provider
  4. Parses the structured response into (refined_prompt, explanation)
"""

import os
from abc import ABC, abstractmethod

import requests

# ──────────────────────────────────────────────────────────────
# System Prompt Template
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert prompt engineer. Your job is to take a user's rough, vague prompt and rewrite it into a highly effective, specific prompt.

You have been given:
1. A PROMPT-ENGINEERING TECHNIQUE to apply
2. A DOMAIN-SPECIFIC BEST PRACTICE relevant to the user's field
3. The USER'S PROFILE (domain, interests, bio) for personalization
4. Optionally, SIMILAR PAST PROMPTS from this user — showing their typical writing patterns

RULES:
- Rewrite the prompt using the technique and domain tip provided.
- Personalize it based on the user's domain, interests, and bio where relevant.
- If similar past prompts are provided, note any recurring weaknesses (e.g. missing audience, no constraints) and proactively fix them in the refined prompt.
- The refined prompt should be significantly more specific, actionable, and effective than the original.
- Do NOT just add generic filler — every addition should serve a clear purpose.

OUTPUT FORMAT (you MUST follow this exactly):
REFINED PROMPT: <your rewritten prompt here — just the prompt text, no quotes>
WHY: <1-2 sentences explaining what specific changes you made and why they improve the prompt>

Do NOT include any other text, headers, or commentary outside this format."""


# ──────────────────────────────────────────────────────────────
# Abstract Base Client
# ──────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Base class for LLM provider clients."""

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the response text."""
        ...


# ──────────────────────────────────────────────────────────────
# Groq Client (default, free tier)
# ──────────────────────────────────────────────────────────────

class GroqClient(LLMClient):
    """LLM client using the Groq Python SDK."""

    def __init__(self):
        from groq import Groq  # Lazy import to avoid errors if not installed

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request via the Groq SDK."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# Ollama Client (fully local fallback)
# ──────────────────────────────────────────────────────────────

class OllamaClient(LLMClient):
    """LLM client using Ollama's local REST API."""

    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request to the Ollama API."""
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,  # Local models can be slow on first load
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def get_llm_client() -> LLMClient:
    """
    Return the configured LLM client based on LLM_PROVIDER env var.

    Defaults to Groq if not specified.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()

    if provider == "groq":
        return GroqClient()
    elif provider == "ollama":
        return OllamaClient()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Must be 'groq' or 'ollama'."
        )


# ──────────────────────────────────────────────────────────────
# Response Parser
# ──────────────────────────────────────────────────────────────

def parse_llm_response(text: str) -> tuple[str, str]:
    """
    Parse the LLM's structured response into (refined_prompt, explanation).

    Expected format:
        REFINED PROMPT: <prompt text>
        WHY: <explanation text>

    Falls back gracefully if the LLM doesn't follow the format exactly.
    """
    refined_prompt = ""
    explanation = ""

    text = text.strip()

    if "REFINED PROMPT:" in text and "WHY:" in text:
        parts = text.split("WHY:", 1)
        refined_prompt = parts[0].replace("REFINED PROMPT:", "").strip()
        explanation = parts[1].strip()
    elif "REFINED PROMPT:" in text:
        refined_prompt = text.split("REFINED PROMPT:", 1)[1].strip()
        explanation = "The prompt was refined using best practices for clarity and specificity."
    else:
        refined_prompt = text
        explanation = "The prompt was refined using best practices for clarity and specificity."

    return refined_prompt, explanation


# ──────────────────────────────────────────────────────────────
# Main Refinement Function
# ──────────────────────────────────────────────────────────────

def refine_prompt(
    raw_prompt: str,
    technique: dict,
    domain_tip: dict,
    user_profile: dict,
    similar_past_prompts: list[str] | None = None,
) -> tuple[str, str]:
    """
    Refine a raw prompt using RAG-retrieved context and an LLM.

    Args:
        raw_prompt:           The user's original, rough prompt.
        technique:            Retrieved prompt-engineering technique (from ChromaDB).
        domain_tip:           Retrieved domain-specific best practice (from ChromaDB).
        user_profile:         The user's stored profile (domain, interests, bio).
        similar_past_prompts: Optional list of the user's past similar prompts,
                              used to identify recurring patterns and personalize further.

    Returns:
        Tuple of (refined_prompt, explanation).
    """
    user_message = _build_user_message(
        raw_prompt, technique, domain_tip, user_profile,
        similar_past_prompts or [],
    )

    client = get_llm_client()
    response_text = client.chat(SYSTEM_PROMPT, user_message)

    return parse_llm_response(response_text)


def _build_user_message(
    raw_prompt: str,
    technique: dict,
    domain_tip: dict,
    user_profile: dict,
    similar_past_prompts: list[str],
) -> str:
    """Assemble the user message with all RAG-retrieved context."""

    interests = ", ".join(user_profile.get("interests", []))
    bio = user_profile.get("bio", "Not provided")

    # Build the past-prompts section only when history exists
    history_section = ""
    if similar_past_prompts:
        formatted = "\n".join(f"  - {p}" for p in similar_past_prompts)
        history_section = f"""
## SIMILAR PAST PROMPTS FROM THIS USER:
(Use these to identify recurring weaknesses or patterns to proactively address)
{formatted}
"""

    return f"""## RAW PROMPT TO REFINE:
{raw_prompt}

## PROMPT-ENGINEERING TECHNIQUE TO APPLY:
**{technique.get('title', 'N/A')}**
When to use: {technique.get('when_to_use', 'N/A')}
How: {technique.get('technique', 'N/A')}
Example before: {technique.get('example_before', 'N/A')}
Example after: {technique.get('example_after', 'N/A')}

## DOMAIN-SPECIFIC BEST PRACTICE:
**{domain_tip.get('title', 'N/A')}**
Guidance: {domain_tip.get('guidance', 'N/A')}

## USER PROFILE:
Domain: {user_profile.get('domain', 'N/A')}
Interests: {interests or 'Not specified'}
Bio: {bio}
{history_section}
Now rewrite the raw prompt following the technique and domain tip above, personalized to this user's profile. Use the exact output format specified in your instructions."""
