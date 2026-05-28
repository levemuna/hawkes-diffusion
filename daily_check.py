"""Daily monitoring run.

For every enabled monitored target:
  1. Fetch up to ``POSTS_PER_TARGET`` recent posts via the data layer (mock
     or BrightData).
  2. Compute the diffusion fingerprint of each post.
  3. Compare to the reference clusters.
  4. Save the analysis to the DB.
  5. If the post is flagged 'coordinated' with margin >= REPLY_MARGIN_THRESHOLD,
     draft a reply and append it to the replies queue (status='pending').

Run manually:
    python daily_check.py

Schedule daily via Windows Task Scheduler (every 24h):
    schtasks /Create /SC DAILY /TN "HawkesDiffusionDaily" /TR ^
        "C:\\Users\\levem\\Projects\\hawkes-diffusion\\.venv\\Scripts\\python.exe ^
         C:\\Users\\levem\\Projects\\hawkes-diffusion\\daily_check.py" /ST 09:00

Or just press the "Run Now" button on the Daily Check page of the dashboard.
"""
from __future__ import annotations

import sys
from datetime import datetime

from core import (
    compute_fingerprint,
    list_targets,
    mark_target_checked,
    nearest_cluster,
    queue_reply,
    save_analysis,
)
from replies import build_evidence, draft_reply
from topics import classify_text
from xpoz_client import fetch_recent_for_target

POSTS_PER_TARGET = 6
REPLY_MARGIN_THRESHOLD = 0.30


def run_daily_check(verbose: bool = True) -> dict:
    """Returns a summary dict."""
    started = datetime.utcnow()
    targets = list_targets(enabled_only=True)
    summary = {
        "started_at": started.isoformat(timespec="seconds") + "Z",
        "targets_checked": 0,
        "posts_analyzed": 0,
        "posts_flagged": 0,
        "replies_drafted": 0,
        "errors": [],
        "per_target": [],
    }

    for tgt in targets:
        target_summary = {
            "target": f"{tgt['kind']}:{tgt['value']}",
            "topic": tgt.get("topic"),
            "analyzed": 0,
            "flagged": 0,
        }
        try:
            posts = fetch_recent_for_target(tgt["kind"], tgt["value"],
                                            limit=POSTS_PER_TARGET)
        except Exception as e:
            summary["errors"].append(f"{tgt['kind']}:{tgt['value']} - {e!r}")
            summary["per_target"].append(target_summary)
            continue

        for post in posts:
            fp = compute_fingerprint(post.event_times, post.engager_account_ages)
            result = nearest_cluster(fp, k=5)
            topic = tgt.get("topic") or classify_text(post.text)
            analysis_id = save_analysis(
                url=post.url, fp=fp, result=result,
                topic=topic, engager_ids=post.engager_ids or None,
            )
            target_summary["analyzed"] += 1
            summary["posts_analyzed"] += 1

            if (
                result["verdict"] == "coordinated"
                and (result.get("confidence_margin") or 0) >= REPLY_MARGIN_THRESHOLD
            ):
                draft = draft_reply(post.text, fp, result)
                if draft:
                    queue_reply(
                        url=post.url,
                        draft_text=draft,
                        analysis_id=analysis_id,
                        evidence=build_evidence(fp, result),
                    )
                    summary["replies_drafted"] += 1
                target_summary["flagged"] += 1
                summary["posts_flagged"] += 1

        mark_target_checked(tgt["id"])
        summary["per_target"].append(target_summary)
        summary["targets_checked"] += 1

        if verbose:
            print(f"  {tgt['kind']:>7} {tgt['value']:<30} "
                  f"analyzed={target_summary['analyzed']:>2} "
                  f"flagged={target_summary['flagged']:>2}")

    summary["finished_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return summary


if __name__ == "__main__":
    print("Running daily check...")
    out = run_daily_check(verbose=True)
    print()
    print(f"Targets checked: {out['targets_checked']}")
    print(f"Posts analyzed: {out['posts_analyzed']}")
    print(f"Posts flagged:  {out['posts_flagged']}")
    print(f"Replies drafted: {out['replies_drafted']}")
    if out["errors"]:
        print(f"Errors: {len(out['errors'])}")
        for e in out["errors"]:
            print(f"  - {e}")
        sys.exit(1)
