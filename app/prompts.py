"""
Prompt templates for the Personalized Prompt Refiner.

This is the single source of truth for every system prompt and user message
template used across all pipelines. Change a prompt here; it propagates
everywhere without touching any logic code.

Structure of each template block:
    ## ROLE & OBJECTIVE     — who the LLM is and what it must do
    ## INJECTED CONTEXT     — RAG-retrieved technique + domain tip
    ## USER PROFILE         — personalization fields
    ## TASK CONSTRAINTS     — hard rules and negative constraints
    ## OUTPUT FORMAT        — machine-parseable, strict schema

Domain personas are stored in DOMAIN_PERSONAS and appended to system prompts
to give each working domain a tailored LLM expert identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Any


# ──────────────────────────────────────────────────────────────
# PromptTemplate Dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """
    A reusable prompt template with a system role and a user message template.

    The ``user_template`` uses Python's ``string.Template`` syntax (``$variable``
    or ``${variable}``).  Call ``.render_user(**kwargs)`` to produce the final
    user message.

    Attributes:
        name:            Human-readable identifier (used in logs).
        system:          The static system-role instruction string.
        user_template:   A ``string.Template``-compatible string for the user turn.
    """

    name: str
    system: str
    user_template: str
    _compiled: Template = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = Template(self.user_template)

    def render_user(self, **kwargs: Any) -> str:
        """
        Render the user message template, substituting all ``$key`` placeholders.

        Missing keys raise ``KeyError``; extra keys are silently ignored.

        Args:
            **kwargs: Variable substitutions matching ``$variable`` placeholders.

        Returns:
            The fully rendered user message string.
        """
        return self._compiled.substitute(kwargs)


# ──────────────────────────────────────────────────────────────
# Domain Personas
# ──────────────────────────────────────────────────────────────

DOMAIN_PERSONAS: dict[str, str] = {
    "dev": (
        "You are a senior software engineer and prompt engineering expert with deep expertise "
        "in software architecture, APIs, DevOps, and developer tooling. "
        "When refining prompts, favour precision, code-centric framing, and explicit constraints "
        "such as language version, framework, and error-handling requirements."
    ),
    "marketing": (
        "You are a CMO-level marketing strategist and copywriter with expertise in brand messaging, "
        "conversion optimisation, SEO, and audience segmentation. "
        "When refining prompts, emphasise target audience clarity, call-to-action sharpness, "
        "tone alignment with brand voice, and measurable outcome framing."
    ),
    "data_analysis": (
        "You are a senior data scientist and analytics expert fluent in statistical analysis, "
        "machine learning, data visualisation, and SQL/Python/R pipelines. "
        "When refining prompts, prioritise reproducibility, dataset specificity, metric definition, "
        "and clear expected output format (table, chart, model performance report)."
    ),
    "education": (
        "You are an expert instructional designer and pedagogy specialist with experience "
        "in curriculum design, learning objectives, and adaptive teaching strategies. "
        "When refining prompts, structure output around learning goals, target learner level, "
        "assessment criteria, and pedagogical method (Bloom's taxonomy, Socratic dialogue, etc.)."
    ),
    "research": (
        "You are a research methodologist and academic writing expert experienced in "
        "systematic reviews, hypothesis formulation, citation standards, and scientific argumentation. "
        "When refining prompts, ensure they specify research scope, methodology, required evidence "
        "standards, and whether the output should be an outline, literature synthesis, or full draft."
    ),
    "creative_writing": (
        "You are a creative director and narrative architect with expertise in storytelling, "
        "genre conventions, character development, and voice. "
        "When refining prompts, enrich them with genre, POV, tone, structural constraint (e.g. three-act), "
        "and any stylistic anchors the writer wants to preserve."
    ),
}

_DEFAULT_PERSONA = (
    "You are an expert prompt engineer with broad knowledge across multiple domains."
)


def get_domain_persona(domain: str) -> str:
    """Return the expert persona instruction for the given domain."""
    return DOMAIN_PERSONAS.get(domain, _DEFAULT_PERSONA)


# ──────────────────────────────────────────────────────────────
# Template 1 — Standard Prompt Refinement
# ──────────────────────────────────────────────────────────────

_REFINE_SYSTEM = """\
## ROLE & OBJECTIVE
$domain_persona

Your primary task is to take a user's rough, vague prompt and rewrite it into a highly
effective, specific, and actionable prompt — fully personalised to the user's profile.

## TASK CONSTRAINTS
- Apply the PROMPT-ENGINEERING TECHNIQUE exactly as described.
- Apply the DOMAIN-SPECIFIC BEST PRACTICE where relevant.
- Calibrate vocabulary and depth to EXPERIENCE LEVEL:
    beginner      → plain language, avoid jargon, briefly explain concepts
    intermediate  → assume working knowledge, moderate technical depth
    expert        → precise technical terminology, skip basics, concise
- Apply TONE PREFERENCE:
    formal        → professional, structured, no contractions
    balanced      → clear and approachable, natural professional voice
    casual        → direct, conversational, first-person OK
- Use OUTPUT FORMAT PREFERENCE in the prompt you write:
    paragraph     → flowing prose instructions
    bullets       → concise bullet points
    structured    → use headers/sections
    code          → code-first output, minimal prose
- If TOOLS are specified, reference relevant ones naturally — never force them.
- If GOAL is specified, ensure the refined prompt serves that larger purpose.
- If AVOID TOPICS are listed, treat them as hard negative constraints.
- If similar past prompts are provided, note recurring weaknesses (e.g. missing audience,
  no constraints) and proactively fix them.
- The refined prompt must be significantly more specific, actionable, and effective.
- Do NOT add generic filler — every addition must serve a clear purpose.
- If PREFERRED LANGUAGE is not English, write the refined prompt in that language.

## OUTPUT FORMAT
You MUST follow this format exactly — no other text, headers, or commentary:

REFINED PROMPT: <your rewritten prompt — just the prompt text, no quotes>
WHY: <1-2 sentences explaining what specific changes you made and why they improve it>
"""

_REFINE_USER = """\
## INJECTED CONTEXT

### PROMPT-ENGINEERING TECHNIQUE TO APPLY:
**$technique_title**
When to use: $technique_when_to_use
How: $technique_how
Example before: $technique_example_before
Example after: $technique_example_after

### DOMAIN-SPECIFIC BEST PRACTICE:
**$domain_tip_title**
Guidance: $domain_tip_guidance

## USER PROFILE
Domain:                    $user_domain
Experience level:          $experience_level  ← calibrate vocabulary and depth
Tone preference:           $tone_preference   ← match this register in the output
Output format preference:  $output_format     ← structure the prompt's expected output
Preferred language:        $preferred_language ← write the refined prompt in this language
Interests:                 $interests
Bio:                       $bio
Goal / Project context:    $goal
Tools & Technologies:      $tools
Avoid topics / constraints: $avoid_topics
$hf_signals_section
## SIMILAR PAST PROMPTS FROM THIS USER:
$history_section

## RAW PROMPT TO REFINE:
$raw_prompt

Now rewrite the raw prompt following the technique, domain tip, and user profile above.
Use the exact OUTPUT FORMAT specified in the system instructions."""

REFINE_TEMPLATE = PromptTemplate(
    name="standard_refine",
    system=_REFINE_SYSTEM,
    user_template=_REFINE_USER,
)


# ──────────────────────────────────────────────────────────────
# Template 2 — Arabic → English Translation
# ──────────────────────────────────────────────────────────────

_TRANSLATE_SYSTEM = """\
## ROLE & OBJECTIVE
You are a bilingual AI assistant fluent in Arabic and English.
Your task is to translate an Arabic prompt into clear, natural English.

## TASK CONSTRAINTS
- Produce a faithful, literal translation — do not add, remove, or interpret meaning.
- Preserve the original intent and all specific technical details.
- Do NOT paraphrase, summarise, or infer.

## OUTPUT FORMAT
Output ONLY the English translation — no preamble, explanation, labels, or quotes.
"""

_TRANSLATE_USER = """\
Translate the following Arabic prompt to English:

$arabic_prompt"""

TRANSLATE_TEMPLATE = PromptTemplate(
    name="arabic_translate",
    system=_TRANSLATE_SYSTEM,
    user_template=_TRANSLATE_USER,
)


# ──────────────────────────────────────────────────────────────
# Template 3 — Arabic Pipeline Refinement
# ──────────────────────────────────────────────────────────────

_ARABIC_REFINE_SYSTEM = """\
## ROLE & OBJECTIVE
$domain_persona

Your task is to take an English translation of an originally Arabic prompt and rewrite it
into a highly effective, specific, professional English prompt.

You have access to:
  1. The original Arabic prompt (for reference and intent verification)
  2. Its English translation (the base text to refine)
  3. A PROMPT-ENGINEERING TECHNIQUE to apply
  4. A DOMAIN-SPECIFIC BEST PRACTICE
  5. The USER'S PROFILE for personalisation
  6. Optional HuggingFace domain signals to guide calibration

## TASK CONSTRAINTS
- Rewrite the translated English prompt using the technique and domain tip.
- Personalise based on the user's domain, interests, tools, and bio.
- The refined prompt must be significantly more specific, actionable, and professional.
- Write entirely in English — this is the desired output language.
- Do NOT add generic filler — every addition must serve a clear purpose.
- Honour all AVOID TOPICS as hard negative constraints.

## OUTPUT FORMAT
You MUST follow this format exactly — no other text:

REFINED PROMPT: <your rewritten prompt here>
WHY: <1-2 sentences explaining what specific changes you made and why>
"""

_ARABIC_REFINE_USER = """\
## ORIGINAL ARABIC PROMPT:
$arabic_prompt

## ENGLISH TRANSLATION:
$translated_english

## INJECTED CONTEXT

### PROMPT-ENGINEERING TECHNIQUE TO APPLY:
**$technique_title**
When to use: $technique_when_to_use
How: $technique_how
Example before: $technique_example_before
Example after: $technique_example_after

### DOMAIN-SPECIFIC BEST PRACTICE:
**$domain_tip_title**
Guidance: $domain_tip_guidance

## USER PROFILE
Domain:                    $user_domain
Experience level:          $experience_level
Tone preference:           $tone_preference
Output format preference:  $output_format
Preferred language:        $preferred_language
Interests:                 $interests
Bio:                       $bio
Goal / Project context:    $goal
Tools & Technologies:      $tools
Avoid topics / constraints: $avoid_topics
$hf_signals_section
## SIMILAR PAST PROMPTS FROM THIS USER (in English):
$history_section

Rewrite the English translation into a polished, professional, domain-specific prompt.
Use the exact OUTPUT FORMAT specified in the system instructions."""

ARABIC_REFINE_TEMPLATE = PromptTemplate(
    name="arabic_refine",
    system=_ARABIC_REFINE_SYSTEM,
    user_template=_ARABIC_REFINE_USER,
)


# ──────────────────────────────────────────────────────────────
# Helper — Build common substitution kwargs
# ──────────────────────────────────────────────────────────────

def build_template_kwargs(
    raw_prompt: str,
    technique: dict,
    domain_tip: dict,
    user_profile: dict,
    similar_past_prompts: list[str],
    hf_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the keyword arguments dict for rendering REFINE_TEMPLATE or
    ARABIC_REFINE_TEMPLATE user messages.

    Args:
        raw_prompt:           The user's original prompt.
        technique:            RAG-retrieved technique metadata dict.
        domain_tip:           RAG-retrieved domain tip metadata dict.
        user_profile:         User profile dict (all v2.1 fields).
        similar_past_prompts: List of past similar prompt strings.
        hf_signals:           Optional dict of signals from HF pipelines
                              (e.g. {"sentiment": "POSITIVE", "summary": "..."}).

    Returns:
        Dict ready to be passed to ``PromptTemplate.render_user(**kwargs)``.
    """
    # User profile fields with safe defaults
    interests = ", ".join(user_profile.get("interests", [])) or "Not specified"
    tools = ", ".join(user_profile.get("tools", [])) or "Not specified"
    avoid_topics = ", ".join(user_profile.get("avoid_topics", [])) or "None"

    # Past-prompt history section
    if similar_past_prompts:
        formatted = "\n".join(f"  - {p}" for p in similar_past_prompts)
        history_section = f"(Use these to identify recurring weaknesses)\n{formatted}"
    else:
        history_section = "(No history yet — treat this as a fresh user.)"

    # HuggingFace signal injection section
    if hf_signals:
        lines = "\n".join(f"  {k}: {v}" for k, v in hf_signals.items())
        hf_signals_section = (
            f"\n## DOMAIN SIGNALS (from specialist model analysis):\n{lines}\n"
            f"Use these signals to further calibrate depth, tone, and framing.\n"
        )
    else:
        hf_signals_section = ""

    domain = user_profile.get("domain", "N/A")

    return {
        "raw_prompt": raw_prompt,
        "domain_persona": get_domain_persona(domain),
        # Technique fields
        "technique_title": technique.get("title", "N/A"),
        "technique_when_to_use": technique.get("when_to_use", "N/A"),
        "technique_how": technique.get("technique", "N/A"),
        "technique_example_before": technique.get("example_before", "N/A"),
        "technique_example_after": technique.get("example_after", "N/A"),
        # Domain tip fields
        "domain_tip_title": domain_tip.get("title", "N/A"),
        "domain_tip_guidance": domain_tip.get("guidance", "N/A"),
        # User profile fields
        "user_domain": domain,
        "experience_level": user_profile.get("experience_level", "intermediate"),
        "tone_preference": user_profile.get("tone_preference", "balanced"),
        "output_format": user_profile.get("output_format_preference", "paragraph"),
        "preferred_language": user_profile.get("preferred_language", "English"),
        "interests": interests,
        "bio": user_profile.get("bio") or "Not provided",
        "goal": user_profile.get("goal") or "Not specified",
        "tools": tools,
        "avoid_topics": avoid_topics,
        # Dynamic sections
        "history_section": history_section,
        "hf_signals_section": hf_signals_section,
    }
