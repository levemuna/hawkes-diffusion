"""Core analytics: Hawkes simulator, fingerprint extraction, SQLite reference DB,
domain layer (monitored targets, replies queue, engager network)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "reference.db"

FEATURE_NAMES = [
    "time_to_peak_hours",
    "burstiness",
    "decay_exponent",
    "peak_velocity_per_min",
    "avg_account_age_days",
]


# ---------- Hawkes simulator ----------

def simulate_hawkes(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    max_events: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Ogata thinning for univariate Hawkes with exponential kernel:
        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    """
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0

    while t < T and len(events) < max_events:
        if events:
            arr = np.asarray(events)
            lam_bar = mu + alpha * float(np.sum(np.exp(-beta * (t - arr))))
        else:
            lam_bar = mu
        if lam_bar <= 0:
            break

        t = t + rng.exponential(1.0 / lam_bar)
        if t >= T:
            break

        if events:
            arr = np.asarray(events)
            lam_t = mu + alpha * float(np.sum(np.exp(-beta * (t - arr))))
        else:
            lam_t = mu

        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)

    return np.asarray(events)


def simulate_organic(T: float = 24 * 3600, seed: int | None = None) -> np.ndarray:
    """Organic viral diffusion: gentle background, slow decay, sustained tail."""
    return simulate_hawkes(mu=5e-4, alpha=6e-4, beta=7.5e-4, T=T, seed=seed)


def simulate_coordinated(T: float = 24 * 3600, seed: int | None = None) -> np.ndarray:
    """Coordinated amplification: rare ignitions, tight bot-burst clusters."""
    return simulate_hawkes(mu=1e-4, alpha=0.4, beta=0.5, T=T, seed=seed)


# ---------- Fingerprint extraction ----------

def compute_fingerprint(
    event_times: np.ndarray,
    account_ages_days: np.ndarray | None = None,
) -> np.ndarray:
    """Extract 5 diffusion features from a cascade."""
    if event_times.size < 5:
        avg = float(np.mean(account_ages_days)) if (
            account_ages_days is not None and account_ages_days.size > 0
        ) else 365.0
        return np.array([0.0, 0.0, 0.0, 0.0, avg])

    et = np.sort(event_times.astype(float))
    et = et - et[0]

    duration = float(et[-1])
    n_bins = max(int(duration / 60.0), 10)
    counts, edges = np.histogram(et, bins=n_bins)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    bin_width = edges[1] - edges[0]
    velocity = counts / bin_width

    peak_idx = int(np.argmax(velocity))
    time_to_peak_hours = bin_centers[peak_idx] / 3600.0

    inter = np.diff(et)
    if inter.size > 1 and (inter.std() + inter.mean()) > 0:
        burstiness = float((inter.std() - inter.mean()) / (inter.std() + inter.mean()))
    else:
        burstiness = 0.0

    decay_exponent = 0.0
    if peak_idx + 3 < len(velocity):
        tail_t = bin_centers[peak_idx + 1 :] - bin_centers[peak_idx]
        tail_v = velocity[peak_idx + 1 :]
        mask = (tail_v > 0) & (tail_t > 0)
        if mask.sum() >= 3:
            try:
                slope, _ = np.polyfit(np.log(tail_t[mask]), np.log(tail_v[mask]), 1)
                decay_exponent = float(-slope)
            except (np.linalg.LinAlgError, ValueError):
                decay_exponent = 0.0

    peak_velocity_per_min = float(velocity[peak_idx] * 60.0)

    if account_ages_days is None or account_ages_days.size == 0:
        avg_age = 365.0
    else:
        avg_age = float(np.mean(account_ages_days))

    return np.array(
        [time_to_peak_hours, burstiness, decay_exponent, peak_velocity_per_min, avg_age]
    )


# ---------- SQLite schema ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    label TEXT NOT NULL CHECK(label IN ('organic', 'coordinated')),
    time_to_peak_hours REAL NOT NULL,
    burstiness REAL NOT NULL,
    decay_exponent REAL NOT NULL,
    peak_velocity_per_min REAL NOT NULL,
    avg_account_age_days REAL NOT NULL,
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_label ON fingerprints(label);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    topic TEXT,
    verdict TEXT NOT NULL,
    organic_dist REAL,
    coordinated_dist REAL,
    confidence_margin REAL,
    time_to_peak_hours REAL,
    burstiness REAL,
    decay_exponent REAL,
    peak_velocity_per_min REAL,
    avg_account_age_days REAL,
    engager_ids TEXT,           -- JSON array of engager ID hashes
    event_times_json TEXT,      -- JSON array of event timestamps (seconds)
    analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_verdict ON analyses(verdict);
CREATE INDEX IF NOT EXISTS idx_analyzed_at ON analyses(analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_topic ON analyses(topic);

CREATE TABLE IF NOT EXISTS monitored_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK(kind IN ('hashtag', 'keyword', 'account')),
    value TEXT NOT NULL,
    topic TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_checked_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(kind, value)
);

CREATE TABLE IF NOT EXISTS replies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    url TEXT NOT NULL,
    draft_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'approved', 'sent', 'dismissed')),
    evidence_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    decided_at TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);
CREATE INDEX IF NOT EXISTS idx_reply_status ON replies(status);
"""


@contextmanager
def db_connection(path: Path = DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    with db_connection(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


# ---------- Reference fingerprints ----------

def insert_fingerprint(
    fp: np.ndarray,
    label: str,
    source: str,
    note: str = "",
    path: Path = DB_PATH,
) -> None:
    with db_connection(path) as conn:
        conn.execute(
            "INSERT INTO fingerprints "
            "(source, label, time_to_peak_hours, burstiness, decay_exponent, "
            "peak_velocity_per_min, avg_account_age_days, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source, label, *map(float, fp), note),
        )
        conn.commit()


def load_reference(
    label: str | None = None,
    path: Path = DB_PATH,
) -> tuple[np.ndarray, list[str]]:
    with db_connection(path) as conn:
        if label is None:
            rows = conn.execute(
                "SELECT label, time_to_peak_hours, burstiness, decay_exponent, "
                "peak_velocity_per_min, avg_account_age_days FROM fingerprints"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT label, time_to_peak_hours, burstiness, decay_exponent, "
                "peak_velocity_per_min, avg_account_age_days "
                "FROM fingerprints WHERE label = ?",
                (label,),
            ).fetchall()
    if not rows:
        return np.zeros((0, 5)), []
    labels = [r["label"] for r in rows]
    feats = np.asarray(
        [[r["time_to_peak_hours"], r["burstiness"], r["decay_exponent"],
          r["peak_velocity_per_min"], r["avg_account_age_days"]] for r in rows],
        dtype=float,
    )
    return feats, labels


def k_nearest_neighbors(
    query: np.ndarray, k: int = 5, label: str = "organic",
    path: Path = DB_PATH,
) -> list[dict]:
    """Return the K nearest reference fingerprints from a given label cluster.

    Each item: {distance, label, raw_features (5-tuple), normalized (5-tuple)}.
    Distances are in the standardized space (same as nearest_cluster uses).
    """
    all_feats, all_labels = load_reference(path=path)
    if all_feats.shape[0] == 0:
        return []
    mu = all_feats.mean(axis=0)
    sd = all_feats.std(axis=0) + 1e-9
    norm_all = (all_feats - mu) / sd
    norm_q = (query - mu) / sd

    mask = np.array([l == label for l in all_labels])
    if mask.sum() == 0:
        return []
    norm_subset = norm_all[mask]
    raw_subset = all_feats[mask]
    dists = np.linalg.norm(norm_subset - norm_q, axis=1)
    order = np.argsort(dists)[:k]

    return [
        {
            "rank": int(rank + 1),
            "distance": float(dists[idx]),
            "label": label,
            "raw_features": tuple(float(x) for x in raw_subset[idx]),
            "normalized": tuple(float(x) for x in norm_subset[idx]),
        }
        for rank, idx in enumerate(order)
    ]


def nearest_cluster(query: np.ndarray, k: int = 5, path: Path = DB_PATH) -> dict:
    all_feats, all_labels = load_reference(path=path)
    if all_feats.shape[0] == 0:
        return {"verdict": "no_reference", "organic_dist": None, "coordinated_dist": None}

    mu = all_feats.mean(axis=0)
    sigma = all_feats.std(axis=0) + 1e-9
    norm_all = (all_feats - mu) / sigma
    norm_q = (query - mu) / sigma

    org_mask = np.array([l == "organic" for l in all_labels])
    coord_mask = np.array([l == "coordinated" for l in all_labels])

    def knn_mean_dist(ref: np.ndarray) -> float:
        if ref.shape[0] == 0:
            return float("inf")
        dists = np.linalg.norm(ref - norm_q, axis=1)
        kk = min(k, dists.size)
        return float(np.sort(dists)[:kk].mean())

    org_dist = knn_mean_dist(norm_all[org_mask])
    coord_dist = knn_mean_dist(norm_all[coord_mask])
    verdict = "organic" if org_dist < coord_dist else "coordinated"
    d_min, d_max = min(org_dist, coord_dist), max(org_dist, coord_dist)
    margin = 1.0 - d_min / (d_max + 1e-9)

    return {
        "verdict": verdict,
        "organic_dist": org_dist,
        "coordinated_dist": coord_dist,
        "confidence_margin": margin,
        "norm_query": norm_q,
        "norm_reference": norm_all,
        "reference_labels": all_labels,
        "feature_mean": mu,
        "feature_std": sigma,
    }


# ---------- Analysis history ----------

def save_analysis(
    url: str,
    fp: np.ndarray,
    result: dict,
    topic: str | None = None,
    engager_ids: list[str] | None = None,
    event_times: np.ndarray | None = None,
    path: Path = DB_PATH,
) -> int:
    import json
    with db_connection(path) as conn:
        cur = conn.execute(
            "INSERT INTO analyses "
            "(url, topic, verdict, organic_dist, coordinated_dist, confidence_margin, "
            "time_to_peak_hours, burstiness, decay_exponent, "
            "peak_velocity_per_min, avg_account_age_days, engager_ids, "
            "event_times_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                url, topic, result["verdict"],
                result.get("organic_dist"), result.get("coordinated_dist"),
                result.get("confidence_margin"),
                *map(float, fp),
                json.dumps(engager_ids) if engager_ids else None,
                json.dumps([float(t) for t in event_times.tolist()])
                if event_times is not None and len(event_times) > 0 else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_analyses(
    limit: int = 200,
    verdict: str | None = None,
    topic: str | None = None,
    path: Path = DB_PATH,
) -> list[dict]:
    query = (
        "SELECT * FROM analyses WHERE 1=1"
    )
    args: list = []
    if verdict:
        query += " AND verdict = ?"
        args.append(verdict)
    if topic:
        query += " AND topic = ?"
        args.append(topic)
    query += " ORDER BY analyzed_at DESC LIMIT ?"
    args.append(limit)
    with db_connection(path) as conn:
        rows = conn.execute(query, args).fetchall()
    return [dict(r) for r in rows]


# ---------- Monitored targets ----------

def add_target(kind: str, value: str, topic: str | None = None,
               path: Path = DB_PATH) -> int:
    with db_connection(path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO monitored_targets (kind, value, topic) "
            "VALUES (?, ?, ?)",
            (kind, value, topic),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_targets(enabled_only: bool = False, path: Path = DB_PATH) -> list[dict]:
    with db_connection(path) as conn:
        if enabled_only:
            rows = conn.execute(
                "SELECT * FROM monitored_targets WHERE enabled = 1 ORDER BY topic, value"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM monitored_targets ORDER BY topic, value"
            ).fetchall()
    return [dict(r) for r in rows]


def set_target_enabled(target_id: int, enabled: bool, path: Path = DB_PATH) -> None:
    with db_connection(path) as conn:
        conn.execute(
            "UPDATE monitored_targets SET enabled = ? WHERE id = ?",
            (1 if enabled else 0, target_id),
        )
        conn.commit()


def remove_target(target_id: int, path: Path = DB_PATH) -> None:
    with db_connection(path) as conn:
        conn.execute("DELETE FROM monitored_targets WHERE id = ?", (target_id,))
        conn.commit()


def mark_target_checked(target_id: int, path: Path = DB_PATH) -> None:
    with db_connection(path) as conn:
        conn.execute(
            "UPDATE monitored_targets SET last_checked_at = CURRENT_TIMESTAMP WHERE id = ?",
            (target_id,),
        )
        conn.commit()


# ---------- Replies queue ----------

def queue_reply(
    url: str,
    draft_text: str,
    analysis_id: int | None = None,
    evidence: dict | None = None,
    path: Path = DB_PATH,
) -> int:
    import json
    with db_connection(path) as conn:
        cur = conn.execute(
            "INSERT INTO replies (analysis_id, url, draft_text, evidence_json) "
            "VALUES (?, ?, ?, ?)",
            (analysis_id, url, draft_text,
             json.dumps(evidence) if evidence else None),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_replies(status: str | None = None, path: Path = DB_PATH) -> list[dict]:
    with db_connection(path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM replies WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM replies ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def update_reply(
    reply_id: int,
    *,
    status: str | None = None,
    draft_text: str | None = None,
    path: Path = DB_PATH,
) -> None:
    set_parts: list[str] = []
    args: list = []
    if status is not None:
        set_parts.append("status = ?")
        args.append(status)
        set_parts.append("decided_at = CURRENT_TIMESTAMP")
    if draft_text is not None:
        set_parts.append("draft_text = ?")
        args.append(draft_text)
    if not set_parts:
        return
    args.append(reply_id)
    with db_connection(path) as conn:
        conn.execute(
            f"UPDATE replies SET {', '.join(set_parts)} WHERE id = ?", args
        )
        conn.commit()


# ---------- Engager network analytics ----------

def engager_network_stats(path: Path = DB_PATH) -> list[dict]:
    """Return list of (engager_id, n_flagged_posts, n_total_posts, suspicion_score)."""
    import json
    from collections import Counter, defaultdict

    with db_connection(path) as conn:
        rows = conn.execute(
            "SELECT verdict, engager_ids FROM analyses WHERE engager_ids IS NOT NULL"
        ).fetchall()

    flagged = Counter()
    total = Counter()
    for r in rows:
        try:
            ids = json.loads(r["engager_ids"])
        except (TypeError, json.JSONDecodeError):
            continue
        for eid in ids:
            total[eid] += 1
            if r["verdict"] == "coordinated":
                flagged[eid] += 1

    out = []
    for eid, n_total in total.items():
        n_flag = flagged.get(eid, 0)
        if n_total < 2:
            continue
        score = n_flag / n_total
        out.append({
            "engager_id": eid,
            "n_flagged": n_flag,
            "n_total": n_total,
            "suspicion_score": score,
        })
    out.sort(key=lambda x: (-x["suspicion_score"], -x["n_flagged"]))
    return out
