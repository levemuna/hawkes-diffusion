"""Populate reference.db with everything the demo needs:

  * 60 organic + 60 coordinated reference fingerprints
  * Default monitored targets (nutrition hashtags / keywords)
  * One full daily-check run so the dashboard has detections + replies

Run once before launching the app:

    python seed_db.py

The initial sweep ALWAYS uses mock data, regardless of the configured
``DATA_SOURCE``. This keeps cold-start fast and predictable (~5 seconds)
and avoids hitting external APIs before the user has even seen the
dashboard. Live data comes in when the user clicks "Run Now" in the
Daily Check section.
"""
from __future__ import annotations

import os

import numpy as np

from core import (
    DB_PATH,
    add_target,
    compute_fingerprint,
    init_db,
    insert_fingerprint,
    simulate_coordinated,
    simulate_organic,
)
from topics import default_targets

N_PER_CLASS = 60


def seed() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()

    rng = np.random.default_rng(42)

    # 1. Reference fingerprints
    for i in range(N_PER_CLASS):
        events = simulate_organic(seed=int(rng.integers(1, 1_000_000)))
        ages = rng.gamma(5.0, 200.0, size=len(events))
        fp = compute_fingerprint(events, ages)
        insert_fingerprint(fp, "organic", "synthetic_hawkes", note=f"organic_{i}")

    for i in range(N_PER_CLASS):
        events = simulate_coordinated(seed=int(rng.integers(1, 1_000_000)))
        ages = rng.gamma(2.0, 30.0, size=len(events))
        fp = compute_fingerprint(events, ages)
        insert_fingerprint(fp, "coordinated", "synthetic_hawkes", note=f"coordinated_{i}")

    # 2. Default targets
    for kind, value, topic in default_targets():
        add_target(kind, value, topic)

    # 3. Initial daily-check sweep — force mock backend so cold-start is fast
    #    and never depends on external services.
    prior_data_source = os.environ.get("DATA_SOURCE")
    os.environ["DATA_SOURCE"] = "mock"
    try:
        from daily_check import run_daily_check
        summary = run_daily_check(verbose=False)
    finally:
        if prior_data_source is None:
            os.environ.pop("DATA_SOURCE", None)
        else:
            os.environ["DATA_SOURCE"] = prior_data_source

    print(f"Seeded {DB_PATH}:")
    print(f"  - {N_PER_CLASS * 2} reference fingerprints")
    print(f"  - {len(default_targets())} monitored targets")
    print(f"  - {summary['posts_analyzed']} posts analyzed in initial sweep (mock)")
    print(f"  - {summary['posts_flagged']} flagged as coordinated")
    print(f"  - {summary['replies_drafted']} reply drafts queued")


if __name__ == "__main__":
    seed()
