# emunulla — Hawkes Diffusion Fingerprinting

> Detecting coordinated nutrition-content amplification on X/Twitter by measuring **how** a post spreads, not **what** it says.

## The wedge

A bot squad amplifying a post leaves a statistical signature in *timing* and *engager profile* that survives no matter what the post says. A keto recipe and a flat-earth meme show the same fingerprint when the same network pushes them.

This project compresses each post's diffusion cascade into a 5-feature Hawkes-process fingerprint, then uses **K-Nearest-Neighbors** (K=5, standardized Euclidean) against reference clusters of known organic and known coordinated cascades to classify it as **organic** or **coordinated** — with a confidence margin.

Output is *organic vs coordinated*. **Not** *true vs false*. A human reviewer decides what to do with the flag.

## Live dashboard

Streamlit Cloud deploy: *(URL goes here once deployed)*

For now, run locally with the quick-start below.

## Quick start

```bash
git clone https://github.com/levemuna/emunulla.git
cd emunulla
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # edit with your credentials, or leave it for mock mode
streamlit run app.py
```

First launch seeds a SQLite reference DB (synthetic Hawkes cascades + initial sweep) — takes ~15 s. The dashboard opens at `http://localhost:8501`.

## The 5 features

| Feature | What it measures | Organic | Coordinated |
|---|---|---|---|
| `time_to_peak_hours` | When peak engagement happened | 1–6 h | minutes, or multi-burst |
| `burstiness` | Goh & Barabási variance of inter-arrival times | ≈ 0 (near-Poisson) | > 0.4 |
| `decay_exponent` | Power-law slope after peak | shallow (long tail) | steep |
| `peak_velocity_per_min` | Max events per minute | < 10 | 20+ |
| `avg_account_age_days` | Median engager account age | > 300 d | < 90 d |

## Data backends

Configurable via `DATA_SOURCE` in `.env`:

- `mock` (default) — deterministic synthetic cascades, no API calls. Use for offline dev and demos.
- `brightdata` — BrightData Datasets REST API, synchronous `/scrape` endpoint.
- `snowflake` — BrightData Data Exchange via Snowflake.
- `scraping_browser` — Playwright over CDP against BrightData's hosted Chrome. Local-only (Playwright not on Streamlit Cloud's free-tier build).

See `.env.example` for the credentials each backend needs.

## Project layout

```
app.py                       Streamlit dashboard (9 sections)
core.py                      Hawkes simulator, KNN classifier, SQLite layer
xpoz_client.py               Data backend router
brightdata_client.py         BrightData REST API
snowflake_client.py          Snowflake backend
scraping_browser_client.py   Playwright + BrightData hosted Chrome
topics.py                    Keyword-based nutrition topic classifier
replies.py                   Reply drafter (Claude API or template fallback)
daily_check.py               Sweep monitored targets; queue reply drafts
seed_db.py                   Populate reference.db on first run
smoke_test.py                Synthetic accuracy check
```

## Dashboard sections

- **Overview** — story, wedge, working numbers
- **Methodology** — KNN reference map, the 5 features in detail
- **Analyze Post** — paste a URL, get fingerprint + verdict + nearest-neighbor evidence
- **Detections** — full corpus, filterable
- **Forensics** — SOC-style panels: diffusion replay vs organic baseline, engager-overlap matrix, peak velocity distribution, topic anomaly week-over-week, feature deviation
- **Topics & Accounts** — topic share, coordinated rate by topic, suspected engager network
- **Targets** — add / toggle hashtags, keywords, accounts to monitor
- **Reply Queue** — human-in-the-loop drafts (nothing auto-posts)
- **Daily Check** — sweep enabled targets; backend smoke-checks; BrightData browser-session monitoring

## Known limitations

- **`brightdata` and `snowflake` backends synthesize the engagement timeline** from aggregate like/repost/reply counts — those datasets don't return per-event timestamps. `burstiness` and `peak_velocity_per_min` features computed on these backends are therefore approximations. For real per-event timestamps, use `scraping_browser` (local-only).
- **No real Twitter posting.** Reply drafts land in the local-DB review queue as `pending`; promoting to `sent` is just a DB flag. Wire `TWITTER_*` env vars and add the posting call if you want auto-send.
- **Topic discovery** depends on the configured dataset. The default BrightData dataset (`gd_lwxkxvnf1cynvib9co`) is URL-input only — `Daily Check → Run Now` returns empty for hashtag/keyword targets unless you also configure a discovery dataset.
- **No automated tests beyond `smoke_test.py`**, which only validates the classifier on synthetic cascades.

## License

Not set. Treat as private until a `LICENSE` file is added.
