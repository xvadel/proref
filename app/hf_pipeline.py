"""
HuggingFace Specialist Pipeline Registry.

Provides a lazy-loaded, cached registry of domain-specific HuggingFace
pipelines used as pre-processors and signal extractors before the main
LLM call. Models are downloaded on first use (~80 MB–900 MB depending
on the model); subsequent calls use the in-memory cached pipeline.

All pipelines are disabled when ``HF_USE_PIPELINE=false`` in .env — every
public function then returns a graceful no-op fallback so the rest of the
codebase never needs to branch on this setting.

Pipeline roles (pre-processor / signal extractor, NOT LLM replacement):
  ┌──────────────────┬──────────────────────────────────────┬──────────────────────────────────────┐
  │ Domain           │ Task                                 │ Default Model                        │
  ├──────────────────┼──────────────────────────────────────┼──────────────────────────────────────┤
  │ Arabic → English │ translation                          │ Helsinki-NLP/opus-mt-ar-en           │
  │ data_analysis    │ sentiment / intent classification    │ distilbert-base-uncased-finetuned-   │
  │                  │                                      │   sst-2-english                      │
  │ marketing        │ financial tone classification        │ ProsusAI/finbert                     │
  │ research         │ summarization (intent condensation)  │ facebook/bart-large-cnn              │
  │ creative_writing │ text-generation (seed enrichment)    │ gpt2                                 │
  │ dev              │ fill-mask (code-context anchor)      │ microsoft/codebert-base              │
  │ education        │ zero-shot classification             │ facebook/bart-large-mnli             │
  └──────────────────┴──────────────────────────────────────┴──────────────────────────────────────┘

Usage:
    from app.hf_pipeline import translate_arabic, extract_domain_signals

    english = translate_arabic("مرحبا")
    signals = extract_domain_signals("I want to analyse churn rates", domain="data_analysis")
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Internal Registry — lazy, keyed by "{task}/{model}"
# ──────────────────────────────────────────────────────────────

_pipeline_registry: dict[str, Any] = {}


def _get_pipeline(task: str, model: str, **kwargs: Any) -> Any:
    """
    Return a cached HuggingFace pipeline, loading it on first use.

    Args:
        task:    HuggingFace task string (e.g. 'translation', 'summarization').
        model:   Model identifier on the HuggingFace Hub.
        **kwargs: Extra keyword arguments forwarded to ``pipeline()``.

    Returns:
        A HuggingFace ``Pipeline`` instance.

    Raises:
        ImportError: If ``transformers`` is not installed.
        RuntimeError: If the model fails to load.
    """
    key = f"{task}/{model}"
    if key not in _pipeline_registry:
        logger.info("Loading HuggingFace pipeline: task=%s model=%s", task, model)
        try:
            from transformers import pipeline  # lazy import

            _pipeline_registry[key] = pipeline(task=task, model=model, **kwargs)
            logger.info("Pipeline loaded successfully: %s", key)
        except Exception as exc:
            logger.error(
                "Failed to load HuggingFace pipeline %s: %s", key, exc, exc_info=True
            )
            raise
    return _pipeline_registry[key]


def _hf_disabled_warning(feature: str) -> None:
    """Log a debug-level notice that HF pipelines are disabled."""
    logger.debug("HF_USE_PIPELINE=false — skipping %s.", feature)


# ──────────────────────────────────────────────────────────────
# Public API — Arabic Translation
# ──────────────────────────────────────────────────────────────

def translate_arabic_hf(arabic_text: str) -> str | None:
    """
    Translate Arabic text to English using the Helsinki-NLP MarianMT model.

    This is the **primary** translation path. The LLM acts as fallback when
    this returns ``None`` (disabled or model error).

    Args:
        arabic_text: The Arabic string to translate.

    Returns:
        Translated English string, or ``None`` if HF pipelines are disabled
        or the translation fails.
    """
    if not settings.hf_use_pipeline:
        _hf_disabled_warning("arabic_translation")
        return None

    try:
        pipe = _get_pipeline("translation", settings.hf_translation_model)
        result = pipe(arabic_text, max_length=512)
        translated: str = result[0]["translation_text"]  # type: ignore[index]
        logger.debug(
            "HF translation: '%s...' → '%s...'",
            arabic_text[:40],
            translated[:40],
        )
        return translated
    except Exception as exc:
        logger.warning(
            "HF Arabic translation failed (%s) — will fall back to LLM.", exc
        )
        return None


# ──────────────────────────────────────────────────────────────
# Public API — Domain Signal Extraction
# ──────────────────────────────────────────────────────────────

def extract_domain_signals(raw_prompt: str, domain: str) -> dict[str, str]:
    """
    Run the domain-appropriate HuggingFace specialist pipeline on the raw
    prompt and return a dict of signals to inject into the LLM context.

    Each domain uses a different model and task:
      - data_analysis    → sentiment/intent label + confidence
      - marketing        → financial tone label + confidence (FinBERT)
      - research         → condensed intent summary (BART)
      - creative_writing → enriched creative seed prefix (GPT-2)
      - dev              → code-context anchor (CodeBERT fill-mask)
      - education        → topic/level classification (BART MNLI zero-shot)

    Args:
        raw_prompt: The user's original prompt text.
        domain:     The user's domain key (e.g. 'dev', 'marketing').

    Returns:
        Dict of signal key → value strings. Empty dict if HF is disabled or
        the model fails — the system degrades gracefully.
    """
    if not settings.hf_use_pipeline:
        _hf_disabled_warning(f"domain_signals for {domain}")
        return {}

    try:
        return _DOMAIN_EXTRACTORS.get(domain, _no_op_extractor)(raw_prompt)
    except Exception as exc:
        logger.warning(
            "HF domain signal extraction failed for domain=%s: %s — continuing without signals.",
            domain,
            exc,
        )
        return {}


# ──────────────────────────────────────────────────────────────
# Domain-Specific Extractors (private)
# ──────────────────────────────────────────────────────────────

def _extract_data_analysis(raw_prompt: str) -> dict[str, str]:
    """Sentiment/intent classification for data_analysis domain."""
    pipe = _get_pipeline(
        "text-classification",
        settings.hf_sentiment_model,
        truncation=True,
    )
    result = pipe(raw_prompt[:512])[0]  # type: ignore[index]
    label: str = result["label"]
    score: float = round(result["score"], 3)
    return {
        "prompt_sentiment": label,
        "sentiment_confidence": str(score),
        "signal_note": (
            "Use sentiment/confidence to calibrate whether the prompt is exploratory "
            "(low confidence) or assertive (high confidence) — adjust framing accordingly."
        ),
    }


def _extract_marketing(raw_prompt: str) -> dict[str, str]:
    """Financial/business tone classification using FinBERT."""
    pipe = _get_pipeline(
        "text-classification",
        settings.hf_finbert_model,
        truncation=True,
    )
    result = pipe(raw_prompt[:512])[0]  # type: ignore[index]
    label: str = result["label"]
    score: float = round(result["score"], 3)
    return {
        "business_tone": label,
        "tone_confidence": str(score),
        "signal_note": (
            "FinBERT detected the business tone of this prompt. "
            "Align the refined prompt's language with this register "
            "(positive/neutral → aspirational copy; negative → problem-solution framing)."
        ),
    }


def _extract_research(raw_prompt: str) -> dict[str, str]:
    """Summarise the research prompt to condense its core intent (BART)."""
    pipe = _get_pipeline(
        "summarization",
        settings.hf_summarization_model,
    )
    # Summarization needs enough tokens but not too many
    truncated = raw_prompt[:1024]
    result = pipe(
        truncated,
        max_length=80,
        min_length=20,
        do_sample=False,
    )[0]  # type: ignore[index]
    summary: str = result["summary_text"]
    return {
        "condensed_intent": summary,
        "signal_note": (
            "This summarised intent reflects the core research question. "
            "Ensure the refined prompt addresses this intent explicitly "
            "with specified methodology, scope, and evidence standards."
        ),
    }


def _extract_creative_writing(raw_prompt: str) -> dict[str, str]:
    """Generate a short creative seed continuation to anchor tone (GPT-2)."""
    pipe = _get_pipeline(
        "text-generation",
        settings.hf_creative_model,
    )
    result = pipe(
        raw_prompt[:200],
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,
        pad_token_id=50256,   # GPT-2 EOS token — suppresses warning
    )[0]  # type: ignore[index]
    generated: str = result["generated_text"]
    # Return only the newly generated part (after the input)
    seed = generated[len(raw_prompt):].strip()
    return {
        "creative_seed": seed[:120],
        "signal_note": (
            "This seed continuation reveals the natural stylistic direction of the prompt. "
            "Use it to anchor or contrast tone choices in the refined version."
        ),
    }


def _extract_dev(raw_prompt: str) -> dict[str, str]:
    """CodeBERT fill-mask to surface implicit code concepts."""
    pipe = _get_pipeline(
        "fill-mask",
        settings.hf_code_model,
    )
    # Insert a mask token near the end of a truncated prompt to surface context
    masked_input = raw_prompt[:200].rstrip() + " <mask>."
    try:
        results = pipe(masked_input, top_k=3)
        top_tokens = [r["token_str"].strip() for r in results[:3]]  # type: ignore[index]
        return {
            "implicit_code_concepts": ", ".join(top_tokens),
            "signal_note": (
                "CodeBERT surfaced these implicit programming concepts from the prompt context. "
                "Consider whether the refined prompt should explicitly reference these "
                "or constrain away from irrelevant ones."
            ),
        }
    except Exception:
        return {}


def _extract_education(raw_prompt: str) -> dict[str, str]:
    """Zero-shot topic and pedagogical level classification."""
    pipe = _get_pipeline(
        "zero-shot-classification",
        settings.hf_zeroshot_model,
    )
    candidate_labels = [
        "beginner-level content",
        "intermediate-level content",
        "advanced-level content",
        "conceptual explanation",
        "practical exercise",
        "assessment / quiz",
        "curriculum design",
    ]
    result = pipe(raw_prompt[:512], candidate_labels=candidate_labels)
    top_label: str = result["labels"][0]  # type: ignore[index]
    top_score: float = round(result["scores"][0], 3)  # type: ignore[index]
    return {
        "detected_pedagogical_type": top_label,
        "classification_confidence": str(top_score),
        "signal_note": (
            "Zero-shot classification detected the educational content type. "
            "Structure the refined prompt's expected output accordingly "
            "(e.g. if 'practical exercise', include concrete activity instructions)."
        ),
    }


def _no_op_extractor(_raw_prompt: str) -> dict[str, str]:
    """Fallback for domains without a specialist extractor."""
    return {}


# ── Domain → Extractor mapping ─────────────────────────────────
_DOMAIN_EXTRACTORS: dict[str, Any] = {
    "data_analysis": _extract_data_analysis,
    "marketing": _extract_marketing,
    "research": _extract_research,
    "creative_writing": _extract_creative_writing,
    "dev": _extract_dev,
    "education": _extract_education,
}
