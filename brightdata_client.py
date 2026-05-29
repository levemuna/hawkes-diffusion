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

import json
import os
import time
from typing import Iterable

import numpy as np
import requests

from xpoz_client import FetchedPost
import core

BRIGHTDATA_API = "https://api.brightdata.com/datasets/v3"
BRIGHTDATA_ROOT = "https://api.brightdata.com"
DEFAULT_TIMEOUT_S = 180
POLL_INTERVAL_S = 4

# Curated cache of pre-scraped real posts. BrightData's on-demand /scrape does
# a *live* scrape that can take 5+ minutes per cold URL — unusable for an
# interactive dashboard. We pre-fetch a set of real posts into this committed
# JSON file (via prewarm.py) so Analyze Post returns real BrightData data
# instantly, both locally and on Streamlit Cloud. A cache hit is real data,
# just fetched ahead of time. Cache misses fall through to a live scrape.
POST_CACHE_PATH = os.path.join(os.path.dirname(__file__), "demo_posts.json")


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


def _parse_jsonl(text: str) -> list[dict]:
    """Parse a JSONL body (one JSON object per line) into dict rows.
    Drops blank lines, unparseable lines, and rows carrying an 'error'."""
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and not obj.get("error"):
            rows.append(obj)
    return rows


def _poll_snapshot(snapshot_id: str, timeout_s: int) -> list[dict]:
    """Poll /snapshot/<id> until rows are ready or timeout. Returns data rows."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(
            f"{BRIGHTDATA_API}/snapshot/{snapshot_id}",
            params={"format": "json"},
            headers=_auth_headers(),
            timeout=30,
        )
        if r.status_code == 200:
            # Snapshot endpoint may return JSON array or JSONL.
            try:
                data = r.json()
                if isinstance(data, list):
                    return [d for d in data
                            if isinstance(d, dict) and not d.get("error")]
            except ValueError:
                pass
            rows = _parse_jsonl(r.text)
            if rows:
                return rows
        elif r.status_code not in (202, 204):
            r.raise_for_status()
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Snapshot {snapshot_id} not ready after {timeout_s}s")


def _scrape(inputs: list[dict], timeout_s: int = DEFAULT_TIMEOUT_S) -> list[dict]:
    """Collect rows for the given inputs.

    Uses the synchronous /scrape endpoint, which usually returns the scraped
    rows directly as JSONL. Under load, BrightData instead queues the job and
    returns ``{"snapshot_id": ...}`` — we detect that and fall back to polling
    /snapshot/<id> so the caller always gets real rows (or a TimeoutError),
    never a bogus stub row.
    """
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
    rows = _parse_jsonl(r.text)

    # Async fallback: a single stub row carrying a snapshot_id (and no post
    # fields) means the job was queued, not completed inline. Poll for it.
    if len(rows) == 1 and rows[0].get("snapshot_id") and "url" not in rows[0]:
        return _poll_snapshot(rows[0]["snapshot_id"], timeout_s)
    return rows


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

    # Cap the synthesized cascade window at 24h. Real engagement concentrates
    # in the first day; using the full post age (which can be years) makes
    # fingerprint features fall far outside the reference cluster scale and
    # produces nonsense verdicts. For posts younger than 24h, use actual age.
    from datetime import datetime, timezone
    CASCADE_WINDOW_S = 24 * 3600
    age_seconds = CASCADE_WINDOW_S
    ts = row.get("date_posted") or row.get("timestamp")
    if ts:
        try:
            posted = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            actual = (datetime.now(timezone.utc) - posted).total_seconds()
            age_seconds = float(min(max(actual, 600.0), CASCADE_WINDOW_S))
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


# ---------- Pre-scraped post cache ----------

def _load_cache() -> dict[str, dict]:
    """Load the curated URL→row cache. Returns {} if missing/unreadable."""
    if not os.path.exists(POST_CACHE_PATH):
        return {}
    try:
        with open(POST_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    try:
        with open(POST_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def cache_put(url: str, row: dict) -> None:
    """Store a real scraped row under its URL for instant future fetches."""
    cache = _load_cache()
    cache[url] = row
    _save_cache(cache)


# ---------- Public API ----------

def is_live() -> bool:
    return IS_LIVE


def cached_urls() -> list[str]:
    """URLs available for instant real-data analysis (pre-scraped)."""
    return list(_load_cache().keys())


def fetch_post(url: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> FetchedPost | None:
    """Fetch a single post by URL via BrightData.

    Returns None on any failure (timeout, network, queued-but-not-ready) so the
    caller in xpoz_client can fall back to mock gracefully instead of crashing
    mid-demo. A cold-cache URL that BrightData queues asynchronously may exceed
    the timeout — pre-warm demo URLs to keep this on the fast inline path.
    """
    if not IS_LIVE:
        return None
    # Cache hit: real data, pre-fetched — return instantly.
    cache = _load_cache()
    if url in cache:
        return _row_to_fetched_post(cache[url])
    # Cache miss: live scrape (may be slow / time out → graceful None).
    try:
        rows = _scrape([{"url": url}], timeout_s=timeout_s)
    except (requests.RequestException, TimeoutError):
        return None
    if not rows:
        return None
    cache_put(url, rows[0])  # remember for next time
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
    try:
        rows = _scrape(inputs, timeout_s=timeout_s)
    except (requests.RequestException, TimeoutError):
        return []
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


def get_browser_session(session_id: str) -> dict:
    """Return full details for one browser session.

    Endpoint: GET /browser_sessions/<id>
    The payload is wrapped as ``{"session": {...}}``; we unwrap it.

    Returns ``{"ok": bool, "session": {...}}`` — does not raise.
    """
    if not API_KEY:
        return {"ok": False, "reason": "BRIGHTDATA_API_KEY not set in .env"}
    if not session_id:
        return {"ok": False, "reason": "No session_id given"}
    try:
        r = requests.get(
            f"{BRIGHTDATA_ROOT}/browser_sessions/{session_id}",
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
        return {"ok": False, "reason": "Non-JSON response", "body": r.text[:300]}

    session = data.get("session", data) if isinstance(data, dict) else {}
    if not isinstance(session, dict):
        session = {}
    return {"ok": True, "session": session}


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
