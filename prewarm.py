"""Pre-scrape a curated set of real posts into demo_posts.json.

BrightData's on-demand /scrape does a *live* scrape that can take several
minutes per cold URL — too slow for an interactive dashboard. Run this once
ahead of a demo (or in CI) to populate the committed cache so Analyze Post
serves real BrightData data instantly.

    python prewarm.py                      # uses the built-in curated list
    python prewarm.py <url> [<url> ...]    # add/refresh specific URLs

Requires DATA_SOURCE=brightdata + a working BRIGHTDATA_API_KEY in .env.
Be patient: a batch of cold URLs can take 5-10 minutes total.
"""
from __future__ import annotations

import sys
import time

import brightdata_client as b

# A spread of real posts: huge verified news/sports accounts (organic-viral),
# plus room to add suspected-amplification examples for contrast.
CURATED_URLS = [
    "https://x.com/CNN/status/1796673270344810776",
    "https://x.com/FabrizioRomano/status/1665296716721946625",
    "https://x.com/FabrizioRomano/status/1683559267524136962",
    "https://x.com/CNN/status/1552015619251634176",
]


def prewarm(urls: list[str], batch_timeout_s: int = 600) -> None:
    if not b.IS_LIVE:
        print("BrightData not live. Set DATA_SOURCE=brightdata and "
              "BRIGHTDATA_API_KEY in .env first.")
        sys.exit(1)

    print(f"Scraping {len(urls)} URL(s) in one batch (up to {batch_timeout_s}s)...")
    t0 = time.time()
    try:
        rows = b._scrape([{"url": u} for u in urls], timeout_s=batch_timeout_s)
    except Exception as e:  # noqa: BLE001 — report and exit, don't traceback
        print(f"Scrape failed after {time.time()-t0:.0f}s: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"Got {len(rows)} row(s) in {time.time()-t0:.0f}s.")
    saved = 0
    for row in rows:
        url = row.get("url") or row.get("post_url")
        if not url:
            continue
        b.cache_put(url, row)
        saved += 1
        eng = (int(row.get("likes") or 0) + int(row.get("reposts") or 0)
               + int(row.get("replies") or 0))
        print(f"  cached {row.get('user_posted','?'):>18} | {eng:>9,} eng | {url}")

    print(f"\nDone. {saved} post(s) cached to {b.POST_CACHE_PATH}")
    print(f"Total cached now: {len(b.cached_urls())}")


if __name__ == "__main__":
    target = sys.argv[1:] or CURATED_URLS
    prewarm(target)
