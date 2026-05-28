"""BrightData Dataset API client for Twitter/X engagement data.

This is wired but not auto-activated. To go live:
    1. Put your credentials in .env (see .env.example):
         BRIGHTDATA_API_KEY=...
         BRIGHTDATA_TWITTER_DATASET_ID=gd_...
         DATA_SOURCE=brightdata
    2. The Streamlit app will then call ``fetch_recent_posts_for_target`` and
       ``fetch_post_engagement`` instead of mock data.

Uses BrightData's synchronous /scrape endpoint — submit inputs, get rows back
in one round-trip. Avoids the snapshot-polling round-trip of /trigger so
interactive UI calls return in seconds instead of minutes.

Docs: https://docs.brightdata.com/api-reference/web-scraper-api/scrape
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import requests

from xpoz_client import FetchedPost
import core

BRIGHTDATA_API = "https://api.brightdata.com/datasets/v3"
BRIGHTDATA_ROOT = "https://api.brightdata.com"
DEFAULT_TIMEOUT_S = 180


def _load_env() -> dict[str, str]:
    """Tiny .env reader so we don't need python-dotenv. Looks for a .env file
    next to this module and falls back to os.environ values."""
    env: dict[str, str] = {}
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    # os.environ wins if set
    for k, v in os.environ.items():
        if k.startswith(("BRIGHTDATA_", "DATA_SOURCE", "ANTHROPIC_", "TWITTER_")):
            env[k] = v
    return env


_env = _load_env()
API_KEY = _env.get("BRIGHTDATA_API_KEY", "")
DATASET_ID = _env.get("BRIGHTDATA_TWITTER_DATASET_ID", "")
DATA_SOURCE = _env.get("DATA_SOURCE", "mock").lower()
IS_LIVE = bool(API_KEY and DATASET_ID and DATA_SOURCE == "brightdata")


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _scrape(inputs: list[dict], timeout_s: int = DEFAULT_TIMEOUT_S) -> list[dict]:
    """Sync collection: POST inputs, get the scraped rows back in one call."""
    r = requests.post(
        f"{BRIGHTDATA_API}/scrape",
        params={
            "dataset_id": DATASET_ID,
            "notify": "false",
            "include_errors": "true",
        },
        headers=_auth_headers(),
        json={"input": inputs},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("data") or data.get("results") or []
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict) and not row.get("error")]


def _row_to_fetched_post(row: dict) -> FetchedPost:
    """Map a BrightData Twitter dataset row to our FetchedPost shape.

    The exact field names in BrightData rows vary by dataset version. Adjust
    the keys below to match what your dataset emits — common ones are listed.
    """
    url = row.get("url") or row.get("post_url") or ""
    text = row.get("description") or row.get("text") or ""
    author = row.get("user_posted") or row.get("author") or ""

    # Engagement timeline: BrightData's Twitter dataset returns aggregate counts
    # (likes, reposts, replies) rather than per-event timestamps. We synthesise
    # a plausible timeline from the counts + post age, using a Hawkes-like
    # back-fill. If your dataset DOES include per-event timestamps, plug them
    # in here directly.
    likes = int(row.get("likes") or 0)
    reposts = int(row.get("reposts") or row.get("retweets") or 0)
    replies = int(row.get("replies") or row.get("comments") or 0)
    total_events = max(likes + reposts + replies, 5)

    # Post age in seconds
    age_seconds = 24 * 3600
    if "date_posted" in row or "timestamp" in row:
        from datetime import datetime, timezone
        ts = row.get("date_posted") or row.get("timestamp")
        try:
            posted = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_seconds = (datetime.now(timezone.utc) - posted).total_seconds()
        except (ValueError, AttributeError):
            pass

    rng = np.random.default_rng(abs(hash(url)) % (2**32))
    times = np.sort(rng.uniform(0, age_seconds, size=total_events))

    # Engager ages: BrightData doesn't give engager-level data by default in
    # the post dataset. Approximate with a distribution.
    ages = rng.gamma(4.0, 100.0, size=total_events)

    return FetchedPost(
        url=url,
        text=text,
        author=author,
        event_times=times,
        engager_account_ages=ages,
        meta={"raw": row, "source": "brightdata"},
    )


# ---------- Public API ----------

def is_live() -> bool:
    return IS_LIVE


def fetch_post(url: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> FetchedPost | None:
    """Fetch a single post by URL via BrightData."""
    if not IS_LIVE:
        return None
    rows = _scrape([{"url": url}], timeout_s=timeout_s)
    if not rows:
        return None
    return _row_to_fetched_post(rows[0])


def fetch_recent_for_target(
    kind: str,
    value: str,
    limit: int = 20,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> list[FetchedPost]:
    """Fetch recent posts matching a monitored target (hashtag/keyword/account)."""
    if not IS_LIVE:
        return []
    if kind == "hashtag":
        inputs = [{"keyword": value, "num_of_posts": limit}]
    elif kind == "keyword":
        inputs = [{"keyword": value, "num_of_posts": limit}]
    elif kind == "account":
        handle = value.lstrip("@")
        inputs = [{"user_name": handle, "num_of_posts": limit}]
    else:
        return []
    rows = _scrape(inputs, timeout_s=timeout_s)
    return [_row_to_fetched_post(r) for r in rows]


def find_similar_posts(url: str, n: int = 5,
                       timeout_s: int = DEFAULT_TIMEOUT_S) -> list[FetchedPost]:
    """BrightData has no native 'similar posts' endpoint for arbitrary URLs.
    Strategy: fetch the input post, extract a hashtag/keyword from its text,
    then call ``fetch_recent_posts_for_target`` on that.
    """
    if not IS_LIVE:
        return []
    input_post = fetch_post(url, timeout_s=timeout_s)
    if input_post is None:
        return []
    # naive hashtag extraction
    tags = [w for w in input_post.text.split() if w.startswith("#")]
    if tags:
        return fetch_recent_for_target("hashtag", tags[0], limit=n,
                                       timeout_s=timeout_s)
    return []


# ---------- Admin / observability ----------

def list_browser_sessions(limit: int = 50, status: str = "all") -> dict:
    """Return recent BrightData browser sessions (live + historical).

    Endpoint: GET /browser_sessions?limit=&status=&sort=timestamp&order=desc
    Auth: same ``BRIGHTDATA_API_KEY`` used by the Datasets API.

    Returns ``{"ok": bool, "sessions": [...], ...}`` — does not raise.
    """
    if not API_KEY:
        return {"ok": False, "reason": "BRIGHTDATA_API_KEY not set in .env"}
    try:
        r = requests.get(
            f"{BRIGHTDATA_ROOT}/browser_sessions",
            params={
                "limit": int(limit),
                "status": status,
                "sort": "timestamp",
                "order": "desc",
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=20,
        )
    except requests.RequestException as e:
        return {"ok": False, "reason": f"Network error: {e!r}"}

    if r.status_code != 200:
        return {"ok": False, "reason": f"HTTP {r.status_code}",
                "body": r.text[:300]}
    try:
        data = r.json()
    except ValueError:
        return {"ok": False, "reason": "Non-JSON response",
                "body": r.text[:300]}

    sessions = data if isinstance(data, list) else data.get("sessions", data)
    if not isinstance(sessions, list):
        sessions = []
    return {"ok": True, "sessions": sessions}


def summarize_sessions(sessions: list[dict]) -> dict:
    """KPIs over a list of session dicts. Tolerant to schema variation."""
    if not sessions:
        return {"total": 0, "running": 0, "ended": 0, "errored": 0,
                "avg_duration_s": 0.0}
    n = len(sessions)
    n_run = sum(1 for s in sessions
                if str(s.get("status", "")).lower() in ("running", "active"))
    n_err = sum(1 for s in sessions
                if str(s.get("status", "")).lower() in ("error", "errored", "failed"))
    n_end = n - n_run - n_err

    durs = []
    for s in sessions:
        for key in ("duration_s", "duration", "duration_ms"):
            v = s.get(key)
            if v is None:
                continue
            try:
                d = float(v)
            except (TypeError, ValueError):
                continue
            if key == "duration_ms":
                d /= 1000.0
            durs.append(d)
            break
    avg_dur = sum(durs) / len(durs) if durs else 0.0

    return {
        "total": n,
        "running": n_run,
        "ended": n_end,
        "errored": n_err,
        "avg_duration_s": round(avg_dur, 1),
    }
