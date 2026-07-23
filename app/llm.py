"""
LLM client module — pluggable adapter for Groq and Ollama backends.

Provider is selected via the LLM_PROVIDER environment variable:
  - "groq"   → Uses the Groq Python SDK (free tier, needs GROQ_API_KEY)
  - "ollama" → Calls Ollama's local REST API (no API key, needs Ollama running)

Key improvements over previous version:
  - All model settings (temperature, max_tokens, timeout) come from app.config.Settings
  - Prompt templates are imported from app.prompts — no strings live here
  - Tenacity-based retry with exponential back-off for transient API errors
  - HuggingFace specialist pipeline used as primary Arabic translator (LLM fallback)
  - Structured logging replaces all print() statements

The main entry points are:
  - refine_prompt()           → standard English pipeline
  - translate_prompt_to_english() → Arabic → English (HF primary, LLM fallback)
  - translate_and_refine()    → Arabic pipeline (translate + refine)
"""

import requests
from abc import ABC, abstractmethod

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from app.config import settings
from app.logging_config import get_logger
from app.prompts import (
    REFINE_TEMPLATE,
    TRANSLATE_TEMPLATE,
    ARABIC_REFINE_TEMPLATE,
    build_template_kwargs,
)

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Retry Decorator Factory
# ──────────────────────────────────────────────────────────────

def _make_retry_decorator():
    """
    Build a tenacity retry decorator using settings from app.config.

    Retries on:
      - requests.Timeout          (Ollama connection timeout)
      - requests.ConnectionError  (Ollama server not reachable)
      - ConnectionError           (generic network errors)

    Groq-specific rate limit / API errors are also caught via broad Exception
    in the inner try/except and re-raised after exhausting retries.
    """
    return retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(
            multiplier=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        retry=retry_if_exception_type(
            (requests.Timeout, requests.ConnectionError, ConnectionError)
        ),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


_retry = _make_retry_decorator()


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

    Args:
        text: Raw LLM response string.

    Returns:
        Tuple of (refined_prompt, explanation).
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
        logger.debug("LLM response missing WHY section — using default explanation.")
    else:
        refined_prompt = text
        explanation = "The prompt was refined using best practices for clarity and specificity."
        logger.warning(
            "LLM response did not follow the expected format — "
            "returning full text as refined prompt."
        )

    return refined_prompt, explanation


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
# Groq Client
# ──────────────────────────────────────────────────────────────

class GroqClient(LLMClient):
    """LLM client using the Groq Python SDK."""

    def __init__(self) -> None:
        from groq import Groq  # lazy import

        if not settings.groq_api_key_is_set:
            raise ValueError(
                "GROQ_API_KEY is not set or is still the placeholder value. "
                "Get a free key at https://console.groq.com and add it to your .env file."
            )
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        logger.info("GroqClient initialised | model=%s", self._model)

    @_retry
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request via the Groq SDK.

        Retries automatically on transient network errors using the
        settings-driven tenacity decorator.
        """
        logger.debug("Groq chat | model=%s | user_chars=%d", self._model, len(user_prompt))
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_seconds,
            )
            content = response.choices[0].message.content
            logger.debug("Groq response received | chars=%d", len(content or ""))
            return content or ""
        except Exception as exc:
            logger.error("Groq API error: %s", exc, exc_info=True)
            raise


# ──────────────────────────────────────────────────────────────
# Ollama Client
# ──────────────────────────────────────────────────────────────

class OllamaClient(LLMClient):
    """LLM client using Ollama's local REST API."""

    def __init__(self) -> None:
        self._host = settings.ollama_host
        self._model = settings.ollama_model
        logger.info("OllamaClient initialised | host=%s | model=%s", self._host, self._model)

    @_retry
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request to the Ollama REST API.

        Retries on Timeout and ConnectionError (Ollama can be slow to respond
        on first model load). Timeout is driven by settings.llm_timeout_seconds.
        """
        logger.debug("Ollama chat | model=%s | user_chars=%d", self._model, len(user_prompt))
        response = requests.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        logger.debug("Ollama response received | chars=%d", len(content))
        return content


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def get_llm_client() -> LLMClient:
    """
    Return the configured LLM client based on the settings.llm_provider value.

    Defaults to Groq if not specified.
    """
    provider = settings.llm_provider

    if provider == "groq":
        return GroqClient()
    elif provider == "ollama":
        return OllamaClient()
    else:
        # This should be caught by the validator in Settings, but guard anyway
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Must be 'groq' or 'ollama'."
        )


# ──────────────────────────────────────────────────────────────
# Standard Refinement Pipeline
# ──────────────────────────────────────────────────────────────

def refine_prompt(
    raw_prompt: str,
    technique: dict,
    domain_tip: dict,
    user_profile: dict,
    similar_past_prompts: list[str] | None = None,
) -> tuple[str, str]:
    """
    Refine a raw prompt using RAG-retrieved context, HF domain signals, and an LLM.

    Pipeline:
      1. Extract domain-specific signals via HuggingFace specialist pipeline.
      2. Build the structured user message from app.prompts templates.
      3. Call the LLM with retry/timeout strategy.
      4. Parse and return (refined_prompt, explanation).

    Args:
        raw_prompt:           The user's original, rough prompt.
        technique:            Retrieved prompt-engineering technique dict.
        domain_tip:           Retrieved domain-specific best practice dict.
        user_profile:         The user's stored profile dict.
        similar_past_prompts: Optional list of the user's past similar prompts.

    Returns:
        Tuple of (refined_prompt, explanation).
    """
    from app.hf_pipeline import extract_domain_signals

    domain = user_profile.get("domain", "")
    logger.info(
        "refine_prompt | user=%s | domain=%s | prompt_chars=%d",
        user_profile.get("user_id", "?"),
        domain,
        len(raw_prompt),
    )

    # Step 1: Extract HuggingFace domain signals (lazy, graceful fallback)
    hf_signals = extract_domain_signals(raw_prompt, domain)
    if hf_signals:
        logger.debug("HF signals extracted for domain=%s: %s", domain, list(hf_signals.keys()))

    # Step 2: Build prompt using the template from app.prompts
    kwargs = build_template_kwargs(
        raw_prompt=raw_prompt,
        technique=technique,
        domain_tip=domain_tip,
        user_profile=user_profile,
        similar_past_prompts=similar_past_prompts or [],
        hf_signals=hf_signals if hf_signals else None,
    )
    system_msg = REFINE_TEMPLATE.system.replace("$domain_persona", kwargs["domain_persona"])
    user_msg = REFINE_TEMPLATE.render_user(**kwargs)

    # Step 3: LLM call (with retry)
    client = get_llm_client()
    response_text = client.chat(system_msg, user_msg)

    # Step 4: Parse response
    refined, explanation = parse_llm_response(response_text)
    logger.info(
        "Refinement complete | refined_chars=%d | explanation_chars=%d",
        len(refined),
        len(explanation),
    )
    return refined, explanation


# ──────────────────────────────────────────────────────────────
# Arabic Translation — HF Primary, LLM Fallback
# ──────────────────────────────────────────────────────────────

def translate_prompt_to_english(arabic_prompt: str) -> str:
    """
    Translate an Arabic prompt to English.

    Strategy:
      1. Try HuggingFace Helsinki-NLP/opus-mt-ar-en (local, free, fast).
      2. If HF is disabled or fails, fall back to the configured LLM.

    Args:
        arabic_prompt: The raw Arabic text to translate.

    Returns:
        English translation string.
    """
    from app.hf_pipeline import translate_arabic_hf

    logger.info("translate_prompt_to_english | chars=%d", len(arabic_prompt))

    # Primary: HuggingFace local model
    hf_result = translate_arabic_hf(arabic_prompt)
    if hf_result:
        logger.info("Translation completed via HuggingFace model.")
        return hf_result

    # Fallback: LLM-based translation
    logger.info("HF translation unavailable — falling back to LLM.")
    client = get_llm_client()
    user_msg = TRANSLATE_TEMPLATE.render_user(arabic_prompt=arabic_prompt)
    result = client.chat(TRANSLATE_TEMPLATE.system, user_msg).strip()
    logger.info("LLM fallback translation complete | chars=%d", len(result))
    return result


# ──────────────────────────────────────────────────────────────
# Arabic Translate + Refine Pipeline
# ──────────────────────────────────────────────────────────────

def translate_and_refine(
    arabic_prompt: str,
    translated_english: str,
    technique: dict,
    domain_tip: dict,
    user_profile: dict,
    similar_past_prompts: list[str] | None = None,
) -> tuple[str, str]:
    """
    Refine the translated English text using RAG-augmented context,
    with reference to the original Arabic prompt.

    Pipeline:
      1. Extract HF domain signals from the English translation.
      2. Build the Arabic-refinement user message from app.prompts templates.
      3. Call the LLM with retry/timeout strategy.
      4. Parse and return (refined_prompt, explanation).

    Args:
        arabic_prompt:        The original Arabic raw prompt.
        translated_english:   The literal translation (from translate_prompt_to_english).
        technique:            Retrieved prompt-engineering technique dict.
        domain_tip:           Retrieved domain-specific best practice dict.
        user_profile:         The user's stored profile dict.
        similar_past_prompts: Optional list of the user's past similar prompts.

    Returns:
        Tuple of (refined_prompt, explanation).
    """
    from app.hf_pipeline import extract_domain_signals

    domain = user_profile.get("domain", "")
    logger.info(
        "translate_and_refine | user=%s | domain=%s",
        user_profile.get("user_id", "?"),
        domain,
    )

    # Step 1: Extract HF signals from the English translation
    hf_signals = extract_domain_signals(translated_english, domain)

    # Step 2: Build the Arabic-refinement prompt
    kwargs = build_template_kwargs(
        raw_prompt=translated_english,  # base text for template
        technique=technique,
        domain_tip=domain_tip,
        user_profile=user_profile,
        similar_past_prompts=similar_past_prompts or [],
        hf_signals=hf_signals if hf_signals else None,
    )
    # Inject Arabic-specific fields
    kwargs["arabic_prompt"] = arabic_prompt
    kwargs["translated_english"] = translated_english

    system_msg = ARABIC_REFINE_TEMPLATE.system.replace(
        "$domain_persona", kwargs["domain_persona"]
    )
    user_msg = ARABIC_REFINE_TEMPLATE.render_user(**kwargs)

    # Step 3: LLM call (with retry)
    client = get_llm_client()
    response_text = client.chat(system_msg, user_msg)

    # Step 4: Parse response
    refined, explanation = parse_llm_response(response_text)
    logger.info(
        "Arabic refinement complete | refined_chars=%d", len(refined)
    )
    return refined, explanation
