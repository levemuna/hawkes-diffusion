"""Smoke test: simulate fresh cascades of each kind and confirm the
nearest_cluster step routes them correctly.
"""
from __future__ import annotations

import numpy as np

from core import (
    compute_fingerprint,
    nearest_cluster,
    simulate_coordinated,
    simulate_organic,
)


def trial(simulator, expected: str, seed: int) -> tuple[bool, dict, np.ndarray]:
    events = simulator(seed=seed)
    if expected == "organic":
        ages = np.random.default_rng(seed).gamma(5.0, 200.0, size=len(events))
    else:
        ages = np.random.default_rng(seed).gamma(2.0, 30.0, size=len(events))
    fp = compute_fingerprint(events, ages)
    result = nearest_cluster(fp, k=5)
    return result["verdict"] == expected, result, fp


def main() -> None:
    n_trials = 20
    rng = np.random.default_rng(123)

    org_correct = 0
    coord_correct = 0
    org_results = []
    coord_results = []

    for _ in range(n_trials):
        seed = int(rng.integers(1, 1_000_000))
        ok, r, fp = trial(simulate_organic, "organic", seed)
        org_correct += int(ok)
        org_results.append((ok, r["organic_dist"], r["coordinated_dist"], fp))

        ok, r, fp = trial(simulate_coordinated, "coordinated", seed)
        coord_correct += int(ok)
        coord_results.append((ok, r["organic_dist"], r["coordinated_dist"], fp))

    print(f"Organic: {org_correct}/{n_trials} routed correctly")
    print(f"Coordinated: {coord_correct}/{n_trials} routed correctly")

    print("\nSample organic trial:")
    ok, od, cd, fp = org_results[0]
    print(f"  verdict_ok={ok}  d(org)={od:.3f}  d(coord)={cd:.3f}")
    print(f"  fingerprint = {dict(zip(['ttp_h','burst','decay','peak_v','age_d'], fp.round(3)))}")

    print("\nSample coordinated trial:")
    ok, od, cd, fp = coord_results[0]
    print(f"  verdict_ok={ok}  d(org)={od:.3f}  d(coord)={cd:.3f}")
    print(f"  fingerprint = {dict(zip(['ttp_h','burst','decay','peak_v','age_d'], fp.round(3)))}")

    accuracy = (org_correct + coord_correct) / (2 * n_trials)
    print(f"\nOverall accuracy on fresh draws: {accuracy*100:.0f}%")


if __name__ == "__main__":
    main()
