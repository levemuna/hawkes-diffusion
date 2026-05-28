"""Data fetching layer.

Selects a backend based on the ``DATA_SOURCE`` env var (or .env):

  - ``mock`` (default): deterministic synthetic cascades based on URL hash.
  - ``brightdata``: real Twitter/X data via brightdata_client.

The mock mode is the demo-friendly default and keeps the app fully working
without any API keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from core import simulate_coordinated, simulate_organic


@dataclass
class FetchedPost:
    url: str
    text: str
    author: str
    event_times: np.ndarray
    engager_account_ages: np.ndarray
    is_input: bool = False
    engager_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _get_data_source() -> str:
    """Resolve DATA_SOURCE freshly each call so .env edits take effect on rerun."""
    # Re-read .env if present
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "DATA_SOURCE":
                    return v.strip().strip('"').strip("'").lower()
    return os.getenv("DATA_SOURCE", "mock").lower()


def get_data_source() -> str:
    return _get_data_source()


def is_mock() -> bool:
    return _get_data_source() not in ("brightdata", "snowflake", "scraping_browser")


# ---------- Mock generator ----------

def _mock_engager_ids(seed: int, n: int) -> list[str]:
    """Stable pseudo-IDs so the same URL produces the same 'engagers'."""
    rng = np.random.default_rng(seed)
    # Reuse a pool of 200 engager IDs so cross-post overlap is meaningful for
    # the engager-network view.
    pool_size = 200
    return [f"engager_{int(i)}" for i in rng.integers(0, pool_size, size=n)]


def _mock_fetch(url: str, rng_seed: int) -> FetchedPost:
    h = abs(hash(url)) % 1000
    coordinated = (h % 2 == 0)
    if coordinated:
        events = simulate_coordinated(seed=h + rng_seed)
        ages = np.random.default_rng(h).gamma(2.0, 30.0, size=len(events))
        text_topic = ["#ozempic miracle results", "#parasite cleanse changed my life",
                       "#detox protocol", "#seedoils are poison",
                       "berberine ozempic alternative"][h % 5]
        label = "coordinated"
        # Coordinated cascades draw from a smaller engager pool — the "bot squad"
        rng = np.random.default_rng(h + rng_seed)
        eid_pool_offset = (h % 5) * 20  # 5 distinct bot squads
        engager_ids = [f"engager_{int(i) + eid_pool_offset}"
                       for i in rng.integers(0, 25, size=len(events))]
    else:
        events = simulate_organic(seed=h + rng_seed)
        ages = np.random.default_rng(h).gamma(5.0, 200.0, size=len(events))
        text_topic = ["#keto recipe", "#intermittentfasting progress",
                       "#plantbased meal prep", "#guthealth tips",
                       "#rawmilk tradition"][h % 5]
        label = "organic"
        engager_ids = _mock_engager_ids(h + rng_seed, len(events))

    return FetchedPost(
        url=url,
        text=f"{text_topic} | [mock {label}] hash={h}",
        author=f"user_{h}",
        event_times=events,
        engager_account_ages=ages,
        engager_ids=engager_ids,
        meta={"mock_label": label},
    )


def _backend():
    """Lazy-load the right backend module based on DATA_SOURCE."""
    ds = _get_data_source()
    if ds == "scraping_browser":
        import scraping_browser_client
        return scraping_browser_client
    if ds == "snowflake":
        import snowflake_client
        return snowflake_client
    if ds == "brightdata":
        import brightdata_client
        return brightdata_client
    return None


def _mock_list(url: str, n_neighbors: int) -> list[FetchedPost]:
    results = [_mock_fetch(url, 0)]
    results[0].is_input = True
    for i in range(n_neighbors):
        results.append(_mock_fetch(f"{url}#neighbor{i}", i + 1))
    return results


def fetch_post_and_neighbors(url: str, n_neighbors: int = 5) -> list[FetchedPost]:
    """Return [input_post, neighbor_1, ..., neighbor_n].

    Routes to Snowflake or BrightData if configured, otherwise returns mocks.
    Falls back to mock gracefully if the live backend returns nothing,
    so the demo never breaks mid-presentation.
    """
    if is_mock():
        return _mock_list(url, n_neighbors)

    backend = _backend()
    if backend is None:
        return _mock_list(url, n_neighbors)

    input_post = backend.fetch_post(url)
    if input_post is None:
        return _mock_list(url, n_neighbors)
    input_post.is_input = True
    neighbors = backend.find_similar_posts(url, n=n_neighbors)
    return [input_post, *neighbors]


def fetch_recent_for_target(kind: str, value: str, limit: int = 10) -> list[FetchedPost]:
    """Used by the daily-check job."""
    if is_mock():
        return [_mock_fetch(f"target://{kind}/{value}#{i}", i) for i in range(limit)]
    backend = _backend()
    if backend is None:
        return []
    return backend.fetch_recent_for_target(kind, value, limit=limit)
