"""Reply drafter.

Generates short, evidence-backed draft replies to posts the system flagged
as coordinated. Drafts are TEMPLATE-based by default so the project runs
with zero external dependencies; if ANTHROPIC_API_KEY is set, we route to
Claude for natural-language drafting.

IMPORTANT: nothing here actually sends a tweet. Drafts land in the
``replies`` table with status='pending' and require a human click in the
Streamlit UI to advance to 'approved' / 'sent'.
"""
from __future__ import annotations

import os
import random

import numpy as np

TEMPLATES = [
    "Independent diffusion analysis of this post shows engagement patterns "
    "inconsistent with organic spread (burstiness {burst:.2f}, peak velocity "
    "{vel:.0f} ev/min, engager-age median {age:.0f}d). Flagged for review.",

    "This thread's spread fingerprint sits {coord_dist:.1f} units from the "
    "organic cluster vs. {org_dist:.1f} from the coordinated cluster — a "
    "{margin:.0%} margin. Indicators: {indicators}.",

    "Heads up: this post shows multiple markers of coordinated amplification "
    "({indicators}). If you have engagement screenshots or context that "
    "contradicts this read, share — happy to re-score.",

    "Hawkes-process fingerprint here matches known {n_ref}-post coordinated "
    "campaigns more closely than typical viral nutrition content. Confidence "
    "margin {margin:.0%}. {indicators}.",
]


def _indicators(fp: np.ndarray) -> list[str]:
    """Return a list of human-readable evidence strings."""
    out = []
    if fp[1] > 0.4:
        out.append("high burstiness (bot-burst pattern)")
    if fp[3] > 25:
        out.append(f"peak velocity {fp[3]:.0f} events/min")
    if fp[4] < 90:
        out.append(f"young engager accounts (avg {fp[4]:.0f}d)")
    if fp[2] > 0.5:
        out.append("sharp post-peak decay")
    if not out:
        out.append("composite fingerprint outside organic range")
    return out


def _template_draft(fp: np.ndarray, result: dict, n_ref: int = 60) -> str:
    template = random.choice(TEMPLATES)
    inds = ", ".join(_indicators(fp))
    return template.format(
        burst=float(fp[1]),
        vel=float(fp[3]),
        age=float(fp[4]),
        coord_dist=float(result.get("coordinated_dist") or 0.0),
        org_dist=float(result.get("organic_dist") or 0.0),
        margin=float(result.get("confidence_margin") or 0.0),
        indicators=inds,
        n_ref=n_ref,
    )


def _llm_draft(post_text: str, fp: np.ndarray, result: dict) -> str | None:
    """Optional Claude-powered drafter. Returns None if not configured."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=key)
        inds = ", ".join(_indicators(fp))
        prompt = (
            "You are reviewing a social media post that an automated detector "
            "flagged as showing patterns of coordinated (non-organic) "
            "amplification. Draft a SHORT reply (under 240 characters, "
            "professional, non-accusatory) that surfaces the evidence and "
            "invites the author to share counter-context. Do NOT make legal "
            "claims. Evidence:\n"
            f"- Post excerpt: {post_text[:200]}\n"
            f"- Indicators: {inds}\n"
            f"- Confidence margin: {result.get('confidence_margin', 0)*100:.0f}%\n"
            "Return ONLY the reply text."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        # type: ignore[attr-defined]
        return msg.content[0].text.strip()
    except Exception:
        return None


def draft_reply(post_text: str, fp: np.ndarray, result: dict) -> str:
    """Pick the best available drafter. Falls back to template if LLM unavailable."""
    if result.get("verdict") != "coordinated":
        return ""
    llm = _llm_draft(post_text, fp, result)
    if llm:
        return llm
    return _template_draft(fp, result)


def build_evidence(fp: np.ndarray, result: dict) -> dict:
    """Snapshot of evidence we store alongside each draft for audit."""
    return {
        "indicators": _indicators(fp),
        "fingerprint": {
            "time_to_peak_hours": float(fp[0]),
            "burstiness": float(fp[1]),
            "decay_exponent": float(fp[2]),
            "peak_velocity_per_min": float(fp[3]),
            "avg_account_age_days": float(fp[4]),
        },
        "organic_dist": result.get("organic_dist"),
        "coordinated_dist": result.get("coordinated_dist"),
        "confidence_margin": result.get("confidence_margin"),
    }
