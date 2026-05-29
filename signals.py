"""Real engagement-authenticity signals.

The Hawkes diffusion fingerprint (core.py) needs per-event timestamps and
engager account ages — data only the Scraping Browser backend provides. The
BrightData REST / Snowflake backends return *aggregate* counts plus rich,
*real* author metadata. This module turns those real fields into transparent
authenticity signals so the live demo is genuinely data-driven instead of
leaning on a synthesized timeline.

Design principle (same as the rest of the project): we **surface signals, not
verdicts**. Each metric is shown with a plain-language reading and a heuristic
flag. A human reviewer decides. Thresholds are deliberately conservative and
labeled as heuristics — there is no trained classifier behind them.

The key insight these signals exploit: a genuinely viral post generates
*conversation* (replies) and draws engagement proportionate to the author's
reach. A coordinated amplification campaign pumps *spread* (reposts/likes)
while generating little organic discussion, often from accounts whose
follower/following structure looks fresh.
"""
from __future__ import annotations

from typing import Any


def _num(row: dict, *keys: str, default: float = 0.0) -> float:
    """First present, numeric-coercible value among keys."""
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return default


def extract_row(meta: dict | None) -> dict | None:
    """Pull the backend's raw post row out of a FetchedPost.meta, tolerant of
    which backend produced it. Returns None for mock (no real row)."""
    if not isinstance(meta, dict):
        return None
    for key in ("raw", "row", "scraped"):
        row = meta.get(key)
        if isinstance(row, dict):
            return row
    return None


def compute_engagement_signals(row: dict | None) -> dict | None:
    """Compute real engagement/authenticity signals from a post row.

    Returns None if the row is missing or has no usable engagement fields.
    All returned numbers are derived purely from real BrightData/Snowflake
    fields — nothing synthesized.
    """
    if not isinstance(row, dict):
        return None

    likes = _num(row, "likes")
    reposts = _num(row, "reposts", "retweets")
    replies = _num(row, "replies", "comments")
    quotes = _num(row, "quotes")
    total_eng = likes + reposts + replies + quotes
    if total_eng <= 0:
        return None

    followers = _num(row, "followers")
    following = _num(row, "following")
    posts_count = _num(row, "posts_count")
    is_verified = bool(row.get("is_verified", False))

    # Engagement composition — the core amplification signature.
    reply_share = replies / total_eng
    repost_share = reposts / total_eng
    like_share = likes / total_eng

    # Reach-normalized engagement (only meaningful if we have follower count).
    eng_per_follower = (total_eng / followers) if followers > 0 else None

    # Account establishment structure (author-level).
    follower_following = (followers / following) if following > 0 else None

    return {
        "raw": {
            "likes": int(likes),
            "reposts": int(reposts),
            "replies": int(replies),
            "quotes": int(quotes),
            "followers": int(followers),
            "following": int(following),
            "posts_count": int(posts_count),
            "is_verified": is_verified,
        },
        "total_engagement": int(total_eng),
        "reply_share": reply_share,
        "repost_share": repost_share,
        "like_share": like_share,
        "eng_per_follower": eng_per_follower,
        "follower_following_ratio": follower_following,
        "is_verified": is_verified,
    }


def signal_flags(sig: dict) -> list[dict]:
    """Turn signals into a list of {level, label, detail} heuristic flags.

    level: "ok" | "watch" | "concern". These are heuristics, not a classifier.
    """
    flags: list[dict] = []

    reply_share = sig["reply_share"]
    repost_share = sig["repost_share"]
    ffr = sig["follower_following_ratio"]
    epf = sig["eng_per_follower"]
    verified = sig["is_verified"]

    # 1. Conversation starvation: lots of spread, almost no replies.
    if reply_share < 0.01:
        flags.append({
            "level": "concern",
            "label": "Minimal conversation",
            "detail": f"Replies are only {reply_share*100:.1f}% of engagement — "
                      "amplification without organic discussion.",
        })
    elif reply_share < 0.03:
        flags.append({
            "level": "watch",
            "label": "Low conversation",
            "detail": f"Replies are {reply_share*100:.1f}% of engagement.",
        })
    else:
        flags.append({
            "level": "ok",
            "label": "Healthy conversation",
            "detail": f"Replies are {reply_share*100:.1f}% of engagement.",
        })

    # 2. Repost-dominated spread (reposts exceed likes).
    if repost_share > 0.50:
        flags.append({
            "level": "concern",
            "label": "Spread-dominated",
            "detail": f"Reposts are {repost_share*100:.0f}% of engagement — "
                      "spread far outpaces likes, a campaign signature.",
        })
    elif repost_share > 0.30:
        flags.append({
            "level": "watch",
            "label": "Repost-heavy",
            "detail": f"Reposts are {repost_share*100:.0f}% of engagement.",
        })

    # 3. Account establishment structure.
    if ffr is not None:
        if ffr < 2.0:
            flags.append({
                "level": "concern",
                "label": "Fresh-account structure",
                "detail": f"Follower/following ratio {ffr:.1f} — "
                          "established accounts skew far higher.",
            })
        elif ffr < 10.0:
            flags.append({
                "level": "watch",
                "label": "Modest account structure",
                "detail": f"Follower/following ratio {ffr:.1f}.",
            })

    # 4. Reach-normalized engagement extremes (only flag the suspicious high end).
    if epf is not None and epf > 0.20:
        flags.append({
            "level": "watch",
            "label": "Engagement exceeds reach",
            "detail": f"Engagement is {epf*100:.0f}% of follower count — "
                      "unusually high; check for inorganic boosting.",
        })

    # 5. Verification (a mild trust signal, not exculpatory).
    if verified:
        flags.append({
            "level": "ok",
            "label": "Verified author",
            "detail": "Author account is verified.",
        })

    return flags


def concern_score(flags: list[dict]) -> float:
    """0..1 composite from flags. Concern=1.0, watch=0.5, ok=0.0, averaged."""
    if not flags:
        return 0.0
    weight = {"concern": 1.0, "watch": 0.5, "ok": 0.0}
    vals = [weight.get(f["level"], 0.0) for f in flags]
    return sum(vals) / len(vals)
