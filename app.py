"""Hawkes Diffusion Fingerprint — multi-section dashboard.

Launch:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Streamlit Cloud secrets → env ----------
# On Streamlit Cloud, credentials live in st.secrets (TOML) instead of .env.
# Mirror them into os.environ BEFORE importing any backend client so that the
# existing env-driven config in xpoz_client / snowflake_client / brightdata_client
# works identically in both environments. Local .env values still win because
# we use setdefault.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(str(_k), str(_v))
except (FileNotFoundError, AttributeError, Exception):
    pass

from core import (
    DB_PATH,
    FEATURE_NAMES,
    add_target,
    compute_fingerprint,
    engager_network_stats,
    ensure_schema_current,
    k_nearest_neighbors,
    list_analyses,
    list_replies,
    list_targets,
    load_reference,
    nearest_cluster,
    queue_reply,
    remove_target,
    save_analysis,
    set_target_enabled,
    update_reply,
)
from replies import build_evidence, draft_reply
from topics import NUTRITION_TOPICS, all_topic_keys, classify_text
from xpoz_client import fetch_post_and_neighbors, get_data_source, is_mock


st.set_page_config(
    page_title="Hawkes Diffusion Fingerprint",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- First-run bootstrap ----------
# Streamlit Cloud uses an ephemeral filesystem; on cold start the reference DB
# won't exist. Seed it once so the dashboard never lands on an empty state.
# If the DB exists but was created by an older revision, auto-migrate the
# schema so INSERTs against newly-added columns don't crash.
if not DB_PATH.exists():
    boot = st.empty()
    with boot.container():
        st.info(
            "First-time setup: seeding the reference DB (synthetic Hawkes "
            "cascades + initial daily-check sweep). Takes ~15 seconds."
        )
        with st.spinner("Bootstrapping..."):
            import seed_db
            seed_db.seed()
    boot.empty()
    st.rerun()
else:
    ensure_schema_current()


# ---------- Sidebar nav ----------
SECTIONS = [
    "Overview",
    "Methodology",
    "Analyze Post",
    "Detections",
    "Forensics",
    "Topics & Accounts",
    "Targets",
    "Reply Queue",
    "Daily Check",
]

with st.sidebar:
    st.title("Hawkes Diffusion")
    st.caption("Detecting coordinated nutrition-content amplification.")

    section = st.radio("Navigate", SECTIONS, label_visibility="collapsed")

    st.markdown("---")
    feats, labels = load_reference()
    n_org = sum(1 for l in labels if l == "organic")
    n_coord = sum(1 for l in labels if l == "coordinated")
    c1, c2 = st.columns(2)
    c1.metric("Organic refs", n_org)
    c2.metric("Coord. refs", n_coord)

    st.markdown("---")
    ds = get_data_source()
    if ds == "scraping_browser":
        st.success("Data source: **Scraping Browser (live)**")
        st.caption("Playwright over CDP via BrightData hosted Chrome.")
    elif ds == "snowflake":
        st.success("Data source: **Snowflake (live)**")
        st.caption("BrightData data exchange via SQL.")
    elif ds == "brightdata":
        st.success("Data source: **BrightData REST (live)**")
    else:
        st.warning("Data source: **MOCK**")
        st.caption("Set `DATA_SOURCE` in `.env` to `scraping_browser`, `snowflake`, or `brightdata` to go live.")


# =========================================================
# Helpers
# =========================================================

VERDICT_COLOR = {"organic": "#2ecc71", "coordinated": "#e74c3c"}


def fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.0f}%"


def analyses_df(verdict=None, topic=None, limit=500) -> pd.DataFrame:
    rows = list_analyses(limit=limit, verdict=verdict, topic=topic)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["analyzed_at"] = pd.to_datetime(df["analyzed_at"])
    return df


# =========================================================
# Section: Overview
# =========================================================

def render_overview() -> None:
    # ---------- 1. The story (emotional hook) ----------
    st.title("When 'everyone is saying this' isn't true")
    st.markdown(
        "> A teenager scrolls X. In 20 minutes she sees fifty accounts "
        "celebrating an extreme diet. To her, this is *what everyone is doing*. "
        "She tries it. She doesn't know it was eight real people and a "
        "coordinated network of fresh accounts pushing on schedule."
    )
    st.markdown(
        "Nutrition is one of the most weaponized verticals on social media — "
        "vulnerable audiences (teenagers, sick people, parents), high monetary "
        "incentives (supplement and alt-health markets), and mostly "
        "speech-protected so platforms don't intervene. **The harm is real "
        "and current tools don't catch it.**"
    )

    st.markdown("---")

    # ---------- 2. The gap ----------
    st.markdown("## The gap")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.error("**Content moderation needs falsity.**\n\n"
                 "'I personally found carnivore diet improved my X' is "
                 "unfalsifiable. Pushed by 200 fresh accounts, it becomes a "
                 "movement. Platforms can't act on legal speech.")
    with g2:
        st.error("**Reports take weeks. Campaigns last days.**\n\n"
                 "Coordinated-inauthentic-behavior reports run through "
                 "platform integrity teams. By the time something is "
                 "actioned, the campaign has moved on.")
    with g3:
        st.error("**Users see counts, not engagers.**\n\n"
                 "'42K likes' looks like 42K people. Nobody surfaces "
                 "*who* engaged — their account age, their other activity, "
                 "their overlap with known networks.")

    st.markdown("---")

    # ---------- 3. Our wedge ----------
    st.markdown("## Our wedge")
    st.success(
        "**We don't judge content. We measure spread.** "
        "A bot squad has a statistical signature in *timing* and "
        "*engager profile* that survives no matter what they say. "
        "A keto recipe and a flat-earth meme show the same fingerprint "
        "when the same network pushes them. Output: organic vs coordinated, "
        "not true vs false. A human reviewer decides what to do with the flag."
    )

    st.markdown("---")

    # ---------- 4. Proof domain — why nutrition ----------
    st.markdown("## Proof domain — why nutrition")
    pd1, pd2 = st.columns([2, 1])
    with pd1:
        st.markdown(
            "We picked **nutrition on X/Twitter** as the wedge to prove this "
            "tech works end-to-end. It is the highest-leverage proving ground "
            "we could find:\n\n"
            "- **Vulnerable audiences** — teenagers, sick people, parents of "
            "sick kids, people with eating disorders.\n"
            "- **Strong monetary incentives** — supplement industry, "
            "alt-health markets, weight-loss products, anti-mainstream-medicine "
            "campaigns.\n"
            "- **Speech-protected** — most claims are opinions or "
            "unfalsifiable. Platforms can't moderate, but bot amplification is "
            "still real harm.\n"
            "- **High volume** — millions of nutrition posts per day, so "
            "patterns are statistically detectable."
        )
    with pd2:
        st.info(
            "**If it works on nutrition, the same pipeline works on:**\n\n"
            "- Financial scams (crypto pumps, ozempic-alt sales)\n"
            "- Political astroturfing\n"
            "- Health misinformation\n"
            "- Defamation campaigns\n"
            "- State-sponsored influence ops"
        )

    st.markdown("---")

    # ---------- 5. Working proof — live numbers ----------
    st.markdown("## Working proof — last 7 days")
    df = analyses_df(limit=2000)
    if df.empty:
        st.info("No analyses yet. Try **Analyze Post** with any URL, or "
                "run **Daily Check → Run Now** to populate the dashboard.")
        return

    now = pd.Timestamp.now("UTC").tz_localize(None)
    last_7 = df[df["analyzed_at"] >= now - pd.Timedelta(days=7)]
    last_24 = df[df["analyzed_at"] >= now - pd.Timedelta(hours=24)]

    n_total_7d = len(last_7)
    n_flag_7d = int((last_7["verdict"] == "coordinated").sum())
    n_flag_24h = int((last_24["verdict"] == "coordinated").sum())
    pending = len(list_replies(status="pending"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Posts analyzed", n_total_7d)
    k2.metric("Coordinated detected", n_flag_7d,
              f"{(n_flag_7d/max(n_total_7d,1))*100:.0f}% of volume")
    k3.metric("Detected last 24h", n_flag_24h)
    k4.metric("Reply drafts queued", pending)

    st.markdown("---")

    # Trend
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Detections over time")
        df_trend = (
            last_7.assign(date=last_7["analyzed_at"].dt.date)
            .groupby(["date", "verdict"]).size().reset_index(name="count")
        )
        if df_trend.empty:
            st.info("No data in the last 7 days.")
        else:
            fig = px.bar(df_trend, x="date", y="count", color="verdict",
                         color_discrete_map=VERDICT_COLOR, height=320)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Top suspected topics (7d)")
        last_7_flagged = last_7[last_7["verdict"] == "coordinated"]
        if last_7_flagged.empty:
            st.info("No flags this week.")
        else:
            top_topic = (
                last_7_flagged["topic"].fillna("other")
                .value_counts().head(5).reset_index()
            )
            top_topic.columns = ["topic", "flags"]
            fig = px.bar(top_topic, x="flags", y="topic", orientation="h",
                         color_discrete_sequence=["#e74c3c"], height=320)
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Most recent flags")
    recent_flagged = df[df["verdict"] == "coordinated"].head(8)
    if recent_flagged.empty:
        st.info("No coordinated detections yet.")
    else:
        cols = ["analyzed_at", "topic", "url", "confidence_margin",
                "burstiness", "peak_velocity_per_min", "avg_account_age_days"]
        st.dataframe(
            recent_flagged[cols],
            use_container_width=True, hide_index=True,
            column_config={
                "analyzed_at": st.column_config.DatetimeColumn("When"),
                "topic": "Topic",
                "url": st.column_config.LinkColumn("URL"),
                "confidence_margin": st.column_config.NumberColumn(
                    "Margin", format="%.0f%%",
                ),
                "burstiness": st.column_config.NumberColumn("Burst", format="%.2f"),
                "peak_velocity_per_min": st.column_config.NumberColumn(
                    "Peak v", format="%.0f"
                ),
                "avg_account_age_days": st.column_config.NumberColumn(
                    "Age (d)", format="%.0f"
                ),
            },
        )


# =========================================================
# Section: Methodology
# =========================================================

def render_methodology() -> None:
    st.title("Methodology")

    st.markdown(
        "## The bet\n"
        "Coordinated amplification campaigns can fake **what a post says** — "
        "wording, emoji, hashtags, even author identity. They cannot fake "
        "**how the post spreads** without leaving a statistical signature. "
        "Bot squads activate in tight time windows. Their accounts skew young. "
        "Their engagement bursts decay faster than organic interest does. "
        "**We never read the post.** We fingerprint the diffusion."
    )

    st.markdown("## Pipeline")
    st.markdown(
        "1. **Ingest** — pull a post's engagement timeline (likes, reposts, "
        "replies with their timestamps) and engager metadata (account age) "
        "from one of the configured backends.\n"
        "2. **Fingerprint** — compress the cascade into a 5-feature vector.\n"
        "3. **KNN match** — standardize against reference clusters of known "
        "organic and known coordinated cascades, then compute mean distance "
        "to the **K=5 nearest neighbors** in each cluster.\n"
        "4. **Verdict** — assign to whichever cluster sits closer. "
        "Margin = `1 − min/max` of the two distances."
    )

    st.markdown("## The 5 features")
    feat_table = pd.DataFrame([
        {"feature": "time_to_peak_hours",
         "what it measures": "When peak engagement happened, hours after post",
         "organic pattern": "1–6 hours",
         "coordinated pattern": "minutes, or multiple bursts"},
        {"feature": "burstiness",
         "what it measures": "Goh & Barabási variance of inter-arrival times",
         "organic pattern": "≈ 0 (near-Poisson)",
         "coordinated pattern": "> 0.4 (heavy clustering)"},
        {"feature": "decay_exponent",
         "what it measures": "Power-law slope of velocity after peak",
         "organic pattern": "shallow (long tail)",
         "coordinated pattern": "steep (sudden drop)"},
        {"feature": "peak_velocity_per_min",
         "what it measures": "Maximum engagement events per minute",
         "organic pattern": "< 10 / min",
         "coordinated pattern": "20+ / min, sometimes 100s"},
        {"feature": "avg_account_age_days",
         "what it measures": "Median engager account age",
         "organic pattern": "> 300 days",
         "coordinated pattern": "< 90 days, often < 30"},
    ])
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    st.markdown("## The reference space")
    st.caption(
        "Each dot is a known cascade. Axes are standardized time-to-peak and "
        "burstiness. The two clusters are visibly separable even on 2 of the "
        "5 dimensions — KNN uses all five at once."
    )
    all_feats, all_labels = load_reference()
    if all_feats.shape[0] > 0:
        mu = all_feats.mean(axis=0)
        sd = all_feats.std(axis=0) + 1e-9
        norm = (all_feats - mu) / sd
        df_ref = pd.DataFrame({
            "time_to_peak (std)": norm[:, 0],
            "burstiness (std)": norm[:, 1],
            "label": all_labels,
        })
        fig = px.scatter(
            df_ref, x="time_to_peak (std)", y="burstiness (std)",
            color="label", color_discrete_map=VERDICT_COLOR,
            opacity=0.7, height=420,
        )
        fig.update_layout(template="plotly_dark",
                          margin=dict(t=10, b=40, l=40, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Why this isn't content moderation")
    st.markdown(
        "Content moderation says: *this post is misinformation*. "
        "It needs a ground-truth judgment on the claim — slow, expensive, "
        "language-specific, ideologically charged.\n\n"
        "Diffusion fingerprinting says: *this post is being pushed by "
        "something that doesn't look like organic interest*. "
        "It is **claim-agnostic and language-agnostic**. "
        "A keto recipe and a flat-earth meme show the same fingerprint when "
        "the same bot squad pushes them.\n\n"
        "The output is not 'true vs false'. It is **organic vs coordinated**. "
        "Action — labeling, deboosting, alerting, reply-drafting — is left to "
        "the human reviewer in the **Reply Queue**."
    )


# =========================================================
# Section: Analyze Post
# =========================================================

def render_analyze() -> None:
    st.title("Analyze Post")
    st.caption(
        "Paste a post URL. We fetch the post (+ neighbors via the data backend), "
        "extract a 5-feature Hawkes-process fingerprint, and locate it against "
        "the reference clusters."
    )

    col_a, col_b = st.columns([4, 1])
    with col_a:
        url = st.text_input(
            "Post URL",
            placeholder="https://twitter.com/user/status/1234567890",
            key="analyze_url",
        )
    with col_b:
        st.write("")
        st.write("")
        go_btn = st.button("Analyze", type="primary", use_container_width=True)

    if go_btn and not url:
        st.warning("Please paste a post URL.")
        return
    if not go_btn:
        return

    with st.spinner("Fetching and analyzing diffusion..."):
        posts = fetch_post_and_neighbors(url, n_neighbors=5)
        input_post = posts[0]
        neighbors = posts[1:]

        input_fp = compute_fingerprint(
            input_post.event_times, input_post.engager_account_ages
        )
        neighbor_fps = [
            compute_fingerprint(p.event_times, p.engager_account_ages)
            for p in neighbors
        ]

        result = nearest_cluster(input_fp, k=5)
        topic = classify_text(input_post.text)

        analysis_id = save_analysis(
            url=url, fp=input_fp, result=result, topic=topic,
            engager_ids=input_post.engager_ids or None,
            event_times=input_post.event_times,
        )
        # If flagged with reasonable confidence, auto-draft a reply
        # (still pending human approval)
        if (result["verdict"] == "coordinated"
                and (result.get("confidence_margin") or 0) >= 0.30):
            draft = draft_reply(input_post.text, input_fp, result)
            if draft:
                queue_reply(url=url, draft_text=draft, analysis_id=analysis_id,
                            evidence=build_evidence(input_fp, result))

    # Verdict banner
    verdict = result["verdict"]
    org_d = result["organic_dist"]
    coord_d = result["coordinated_dist"]
    margin = result.get("confidence_margin", 0.0) or 0.0
    if verdict == "coordinated":
        st.error(
            f"### Coordinated amplification likely\n"
            f"Topic: **{topic}** · margin **{margin*100:.0f}%** · "
            f"d(coord)=**{coord_d:.2f}** vs d(org)=**{org_d:.2f}**"
        )
    else:
        st.success(
            f"### Organic diffusion likely\n"
            f"Topic: **{topic}** · margin **{margin*100:.0f}%** · "
            f"d(org)=**{org_d:.2f}** vs d(coord)=**{coord_d:.2f}**"
        )

    sub_map, sub_feat, sub_knn, sub_rel = st.tabs(
        ["Fingerprint Map", "Feature Breakdown", "K Nearest Neighbors", "Related Posts"]
    )

    with sub_map:
        norm_ref = result["norm_reference"]
        norm_q = result["norm_query"]
        ref_labels = result["reference_labels"]
        df_ref = pd.DataFrame({
            "time_to_peak (std)": norm_ref[:, 0],
            "burstiness (std)": norm_ref[:, 1],
            "label": ref_labels,
        })
        fig = px.scatter(
            df_ref, x="time_to_peak (std)", y="burstiness (std)",
            color="label", color_discrete_map=VERDICT_COLOR,
            opacity=0.6, height=500,
        )
        fig.add_trace(go.Scatter(
            x=[norm_q[0]], y=[norm_q[1]], mode="markers",
            marker=dict(size=22, color="gold", symbol="star",
                        line=dict(color="black", width=2)),
            name="Your post",
        ))
        st.plotly_chart(fig, use_container_width=True)

    with sub_feat:
        all_feats, all_labels = load_reference()
        org_mean = all_feats[[l == "organic" for l in all_labels]].mean(axis=0)
        coord_mean = all_feats[[l == "coordinated" for l in all_labels]].mean(axis=0)
        df_cmp = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "your post": input_fp,
            "organic mean": org_mean,
            "coordinated mean": coord_mean,
        })
        fig2 = px.bar(
            df_cmp.melt(id_vars="feature", var_name="series", value_name="value"),
            x="feature", y="value", color="series", barmode="group",
            color_discrete_map={
                "your post": "#f1c40f",
                "organic mean": "#2ecc71",
                "coordinated mean": "#e74c3c",
            },
            height=420,
        )
        fig2.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    with sub_knn:
        st.subheader("K nearest reference cascades")
        st.caption(
            "These are the **actual reference fingerprints** the KNN matched "
            "your post against. Distance is in the standardized 5-feature "
            "space. The verdict is whichever cluster's K=5 mean distance is "
            "smaller — the side with closer green or closer red rows wins."
        )
        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown("**Closest organic neighbors**")
            org_nn = k_nearest_neighbors(input_fp, k=5, label="organic")
            if org_nn:
                df_org = pd.DataFrame([
                    {"#": nn["rank"], "distance": round(nn["distance"], 3),
                     **dict(zip(FEATURE_NAMES, nn["raw_features"]))}
                    for nn in org_nn
                ])
                st.dataframe(df_org, use_container_width=True, hide_index=True)
                st.caption(f"Mean distance: **{np.mean([nn['distance'] for nn in org_nn]):.3f}**")
        with kc2:
            st.markdown("**Closest coordinated neighbors**")
            coord_nn = k_nearest_neighbors(input_fp, k=5, label="coordinated")
            if coord_nn:
                df_coord = pd.DataFrame([
                    {"#": nn["rank"], "distance": round(nn["distance"], 3),
                     **dict(zip(FEATURE_NAMES, nn["raw_features"]))}
                    for nn in coord_nn
                ])
                st.dataframe(df_coord, use_container_width=True, hide_index=True)
                st.caption(f"Mean distance: **{np.mean([nn['distance'] for nn in coord_nn]):.3f}**")

        # Distance comparison bar
        if org_nn and coord_nn:
            df_dists = pd.DataFrame({
                "rank": list(range(1, 6)) * 2,
                "distance": [nn["distance"] for nn in org_nn] +
                            [nn["distance"] for nn in coord_nn],
                "cluster": ["organic"] * 5 + ["coordinated"] * 5,
            })
            fig = px.bar(df_dists, x="rank", y="distance", color="cluster",
                         barmode="group", color_discrete_map=VERDICT_COLOR,
                         height=300, title="Distance to each of the 5 nearest neighbors")
            fig.update_layout(template="plotly_dark",
                              margin=dict(t=40, b=40, l=40, r=10))
            st.plotly_chart(fig, use_container_width=True)

    with sub_rel:
        rows = []
        for p, fp in zip(neighbors, neighbor_fps):
            rows.append({
                "URL": p.url,
                "Author": p.author,
                "Topic": classify_text(p.text),
                "Events": int(p.event_times.size),
                "Time-to-peak (h)": round(float(fp[0]), 2),
                "Burstiness": round(float(fp[1]), 3),
                "Peak v (ev/min)": round(float(fp[3]), 2),
                "Avg engager age (d)": round(float(fp[4]), 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =========================================================
# Section: Detections
# =========================================================

def render_detections() -> None:
    st.title("Detections")
    st.caption(
        "Every analyzed post lands here. Filter, sort, drill into any row."
    )

    df = analyses_df(limit=2000)
    if df.empty:
        st.info("No analyses yet.")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        verdict_pick = st.selectbox(
            "Verdict", ["All", "coordinated", "organic"], index=0
        )
    with f2:
        topic_options = ["All"] + sorted(df["topic"].dropna().unique().tolist())
        topic_pick = st.selectbox("Topic", topic_options, index=0)
    with f3:
        min_margin = st.slider("Min confidence margin", 0.0, 1.0, 0.0, 0.05)

    view = df.copy()
    if verdict_pick != "All":
        view = view[view["verdict"] == verdict_pick]
    if topic_pick != "All":
        view = view[view["topic"] == topic_pick]
    view = view[view["confidence_margin"].fillna(0) >= min_margin]

    st.write(f"Showing **{len(view)}** of {len(df)} analyses.")

    cols = ["analyzed_at", "verdict", "topic", "url", "confidence_margin",
            "time_to_peak_hours", "burstiness", "peak_velocity_per_min",
            "avg_account_age_days"]
    st.dataframe(
        view[cols], use_container_width=True, hide_index=True,
        column_config={
            "analyzed_at": st.column_config.DatetimeColumn("When"),
            "verdict": "Verdict",
            "topic": "Topic",
            "url": st.column_config.LinkColumn("URL"),
            "confidence_margin": st.column_config.NumberColumn("Margin", format="%.0f%%"),
            "time_to_peak_hours": st.column_config.NumberColumn("Peak (h)", format="%.2f"),
            "burstiness": st.column_config.NumberColumn("Burst", format="%.2f"),
            "peak_velocity_per_min": st.column_config.NumberColumn("Peak v", format="%.0f"),
            "avg_account_age_days": st.column_config.NumberColumn("Age (d)", format="%.0f"),
        },
    )

    st.markdown("---")
    st.subheader("Drill into a single detection")
    if not view.empty:
        labels = [f"{r['analyzed_at']:%m-%d %H:%M} · {r['verdict']} · {r['url'][:60]}"
                  for _, r in view.iterrows()]
        choice = st.selectbox("Pick a row", labels)
        if choice:
            idx = labels.index(choice)
            row = view.iloc[idx]
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**URL**")
                st.code(row["url"])
                st.markdown(f"**Topic**: `{row['topic']}`  ·  "
                            f"**Verdict**: `{row['verdict']}`  ·  "
                            f"**Margin**: `{fmt_pct(row['confidence_margin'])}`")
            with d2:
                fp_dict = {n: float(row[n]) for n in FEATURE_NAMES}
                st.markdown("**Fingerprint**")
                st.json(fp_dict, expanded=False)


# =========================================================
# Section: Forensics — SOC-style queries
# =========================================================

def _diffusion_curve(event_times: np.ndarray, bin_minutes: int = 5) -> pd.DataFrame:
    """Bin event timestamps into minutes and return a velocity time series."""
    if event_times.size == 0:
        return pd.DataFrame({"minute": [], "events": []})
    et = np.asarray(event_times, dtype=float)
    et = et - et.min()
    bin_s = bin_minutes * 60
    n_bins = max(int(et.max() / bin_s) + 1, 5)
    counts, edges = np.histogram(et, bins=n_bins, range=(0, n_bins * bin_s))
    return pd.DataFrame({
        "minute": edges[:-1] / 60.0,
        "events": counts,
    })


def _expected_organic_curve(duration_minutes: float, total_events: int) -> pd.DataFrame:
    """Synthesize what an organic diffusion of similar size SHOULD look like.
    Uses a log-normal-ish shape (gentle rise, long tail) scaled to total events."""
    n = max(int(duration_minutes / 5), 10)
    t = np.linspace(0.01, duration_minutes, n)
    # Long-tail organic shape: rises to ~10% of duration then decays slowly
    peak_t = duration_minutes * 0.15
    sigma = duration_minutes * 0.5
    shape = np.exp(-((np.log(t) - np.log(peak_t)) ** 2) / (2 * (sigma / duration_minutes) ** 2))
    shape = shape / shape.sum() * total_events
    return pd.DataFrame({"minute": t, "events": shape})


def render_forensics() -> None:
    st.title("Forensics")
    st.caption(
        "SOC-style queries on the detection corpus: cross-post correlation, "
        "campaign clusters, deviation-from-baseline, topic anomalies. "
        "Pick a saved detection on the right to drill into its diffusion curve."
    )

    df = analyses_df(limit=2000)
    if df.empty:
        st.info("No analyses yet. Use **Analyze Post** or **Daily Check → Run Now** "
                "to populate the corpus.")
        return

    # ---------- Top filter bar ----------
    fb1, fb2, fb3, fb4 = st.columns([1, 1, 1, 1])
    with fb1:
        time_window = st.selectbox("Window", ["24h", "7d", "30d", "all"], index=1)
    with fb2:
        only_flagged = st.toggle("Coordinated only", value=True)
    with fb3:
        min_margin = st.slider("Min margin", 0.0, 1.0, 0.0, 0.05)
    with fb4:
        topics_in_data = sorted(df["topic"].dropna().unique().tolist())
        topic_filter = st.multiselect("Topics", topics_in_data, default=[])

    now = pd.Timestamp.now("UTC").tz_localize(None)
    if time_window == "24h":
        df = df[df["analyzed_at"] >= now - pd.Timedelta(hours=24)]
    elif time_window == "7d":
        df = df[df["analyzed_at"] >= now - pd.Timedelta(days=7)]
    elif time_window == "30d":
        df = df[df["analyzed_at"] >= now - pd.Timedelta(days=30)]
    if only_flagged:
        df = df[df["verdict"] == "coordinated"]
    df = df[df["confidence_margin"].fillna(0) >= min_margin]
    if topic_filter:
        df = df[df["topic"].isin(topic_filter)]

    st.write(f"**{len(df)}** detections in scope.")
    if df.empty:
        st.warning("No detections match the current filters.")
        return

    st.markdown("---")

    # ---------- Panel 1: Diffusion curve replay vs organic baseline ----------
    st.subheader("1. Diffusion curve vs organic baseline")
    st.caption(
        "Pick a detection and overlay its real engagement timeline against what "
        "an organic post of the same size would look like. Sharp spikes / multi-burst "
        "patterns / clipped tails are coordinated indicators."
    )
    df_with_events = df[df["event_times_json"].notna()].copy()
    if df_with_events.empty:
        st.info("No detections in scope have replayable event timelines yet. "
                "Run **Analyze Post** or **Daily Check** to generate some.")
    else:
        labels = [
            f"#{r['id']} · {r['analyzed_at']:%m-%d %H:%M} · {r['topic']} · "
            f"margin {(r['confidence_margin'] or 0)*100:.0f}% · {r['url'][:55]}"
            for _, r in df_with_events.iterrows()
        ]
        pick_idx = st.selectbox("Detection", range(len(labels)),
                                format_func=lambda i: labels[i])
        row = df_with_events.iloc[pick_idx]
        try:
            et = np.array(json.loads(row["event_times_json"]), dtype=float)
        except (TypeError, json.JSONDecodeError):
            et = np.array([])
        curve = _diffusion_curve(et, bin_minutes=5)
        total = int(curve["events"].sum()) if not curve.empty else 0
        dur_min = float(curve["minute"].max()) if not curve.empty else 60.0
        baseline = _expected_organic_curve(dur_min, total)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=baseline["minute"], y=baseline["events"],
            name="Expected organic", mode="lines",
            line=dict(color="#2ecc71", width=2, dash="dot"),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.10)",
        ))
        fig.add_trace(go.Bar(
            x=curve["minute"], y=curve["events"], name="Actual engagement",
            marker=dict(color="#e74c3c"),
        ))
        fig.update_layout(
            template="plotly_dark",
            height=380,
            xaxis_title="minutes since post",
            yaxis_title="engagement events / 5-min bin",
            barmode="overlay",
            margin=dict(t=20, b=40, l=40, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drift summary
        actual_peak = curve["events"].max() if not curve.empty else 0
        expected_peak = baseline["events"].max() if not baseline.empty else 0
        peak_ratio = actual_peak / max(expected_peak, 0.5)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Events", total)
        m2.metric("Actual peak", f"{actual_peak:.0f}")
        m3.metric("Expected peak", f"{expected_peak:.0f}")
        m4.metric("Peak ratio", f"{peak_ratio:.1f}x",
                  delta="anomalous" if peak_ratio > 2.5 else "within range",
                  delta_color="inverse")

    st.markdown("---")

    # ---------- Panel 2: Campaign clusters (engager overlap matrix) ----------
    st.subheader("2. Campaign clusters — engager overlap matrix")
    st.caption(
        "Rows and columns are flagged posts. A bright cell at (A, B) means the "
        "same engagers showed up on both posts. Dense blocks = coordinated campaigns "
        "operating through the same bot squad."
    )
    with_engagers = df[df["engager_ids"].notna()].copy()
    if len(with_engagers) < 2:
        st.info("Need at least 2 detections with engager data to compute overlap.")
    else:
        post_engagers = {}
        for _, r in with_engagers.iterrows():
            try:
                eids = json.loads(r["engager_ids"])
            except (TypeError, json.JSONDecodeError):
                continue
            post_engagers[r["id"]] = set(eids)
        ids = list(post_engagers.keys())[:25]  # cap for readable matrix
        n = len(ids)
        matrix = np.zeros((n, n), dtype=int)
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i == j:
                    matrix[i, j] = len(post_engagers[a])
                else:
                    matrix[i, j] = len(post_engagers[a] & post_engagers[b])
        # Mask diagonal for visualization (self-overlap dominates)
        display = matrix.copy().astype(float)
        np.fill_diagonal(display, np.nan)
        labels = [f"#{i}" for i in ids]
        fig = px.imshow(
            display, x=labels, y=labels,
            color_continuous_scale="reds",
            aspect="auto", height=500,
            labels=dict(color="shared engagers"),
        )
        fig.update_layout(template="plotly_dark",
                          margin=dict(t=20, b=40, l=40, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Top overlaps as a ranked table
        overlaps = []
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if j > i and matrix[i, j] > 0:
                    overlaps.append({
                        "post A": a, "post B": b,
                        "shared engagers": int(matrix[i, j]),
                        "A url": with_engagers[with_engagers["id"]==a]["url"].iat[0],
                        "B url": with_engagers[with_engagers["id"]==b]["url"].iat[0],
                    })
        if overlaps:
            df_o = pd.DataFrame(overlaps).sort_values(
                "shared engagers", ascending=False).head(15)
            st.write("**Top shared-engager pairs** (probable same campaign):")
            st.dataframe(df_o, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---------- Panel 3: Peak-velocity distribution ----------
    st.subheader("3. Peak velocity — flagged vs organic baseline")
    st.caption(
        "Density of peak engagement velocity. Organic content concentrates "
        "below ~10 events/min; flagged content shows a long tail to the right "
        "or a separate mode."
    )
    full = analyses_df(limit=5000)
    if not full.empty:
        v_flag = full[full["verdict"] == "coordinated"]["peak_velocity_per_min"].dropna()
        v_org = full[full["verdict"] == "organic"]["peak_velocity_per_min"].dropna()
        fig = go.Figure()
        if len(v_org) > 1:
            fig.add_trace(go.Histogram(x=v_org, name="organic", opacity=0.7,
                                        marker_color="#2ecc71", nbinsx=40))
        if len(v_flag) > 1:
            fig.add_trace(go.Histogram(x=v_flag, name="coordinated", opacity=0.7,
                                        marker_color="#e74c3c", nbinsx=40))
        fig.update_layout(template="plotly_dark", barmode="overlay", height=320,
                          xaxis_title="peak velocity (events/min)",
                          yaxis_title="count",
                          margin=dict(t=10, b=40, l=40, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------- Panel 4: Topic anomaly watch ----------
    st.subheader("4. Topic anomaly watch — coord-rate jump week-over-week")
    st.caption(
        "Topics whose coordinated-detection rate changed most between the "
        "previous 7 days and the current 7 days. Big positive deltas = "
        "a new campaign starting in that vertical."
    )
    full = analyses_df(limit=5000)
    if not full.empty:
        now = pd.Timestamp.now("UTC").tz_localize(None)
        prev = full[(full["analyzed_at"] >= now - pd.Timedelta(days=14)) &
                    (full["analyzed_at"] < now - pd.Timedelta(days=7))]
        curr = full[full["analyzed_at"] >= now - pd.Timedelta(days=7)]

        def rate(slice_df):
            return (slice_df.assign(is_c=(slice_df["verdict"]=="coordinated").astype(int))
                    .groupby(slice_df["topic"].fillna("other"))
                    .agg(rate=("is_c", "mean"), n=("is_c", "size")))
        r_prev, r_curr = rate(prev), rate(curr)
        merged = r_curr.join(r_prev, lsuffix="_curr", rsuffix="_prev", how="outer").fillna(0)
        merged["delta"] = merged["rate_curr"] - merged["rate_prev"]
        merged = merged[merged["n_curr"] >= 2].sort_values("delta", ascending=False)
        if not merged.empty:
            display = merged.reset_index().rename(columns={
                "topic": "Topic", "rate_curr": "rate (7d)", "rate_prev": "rate (prev 7d)",
                "delta": "Δ", "n_curr": "n posts (7d)"})
            fig = px.bar(display, x="Topic", y="Δ",
                         color="Δ", color_continuous_scale="RdYlGn_r",
                         height=320, hover_data=["rate (7d)", "rate (prev 7d)"])
            fig.update_layout(template="plotly_dark", yaxis_tickformat=".0%",
                              margin=dict(t=10, b=40, l=40, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(display[["Topic","rate (7d)","rate (prev 7d)","Δ","n posts (7d)"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("Not enough cross-week volume yet for anomaly detection.")

    st.markdown("---")

    # ---------- Panel 5: Feature deviation radar ----------
    st.subheader("5. Feature deviation — which dimensions are most off")
    st.caption(
        "For each flagged detection, by how many standard deviations it sits "
        "from the organic-cluster mean on each of the 5 fingerprint features. "
        "High values on burstiness + peak_velocity + low avg_account_age "
        "= classic coordinated profile."
    )
    if not df.empty:
        all_feats, all_labels = load_reference()
        org = all_feats[[l == "organic" for l in all_labels]]
        if org.shape[0] > 0:
            mu = org.mean(axis=0)
            sd = org.std(axis=0) + 1e-9
            zs = []
            for _, r in df.head(50).iterrows():
                v = np.array([
                    r["time_to_peak_hours"], r["burstiness"], r["decay_exponent"],
                    r["peak_velocity_per_min"], r["avg_account_age_days"],
                ], dtype=float)
                zs.append(np.abs((v - mu) / sd))
            if zs:
                z_arr = np.vstack(zs)
                df_z = pd.DataFrame(z_arr.mean(axis=0).reshape(1, -1),
                                    columns=FEATURE_NAMES,
                                    index=["mean |z| of flagged in scope"]).T.reset_index()
                df_z.columns = ["feature", "mean |z|"]
                fig = px.bar(df_z, x="feature", y="mean |z|",
                             color="mean |z|", color_continuous_scale="reds",
                             height=300)
                fig.update_layout(template="plotly_dark", xaxis_tickangle=-25,
                                  margin=dict(t=10, b=40, l=40, r=10))
                st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Section: Topics & Accounts
# =========================================================

def render_topics_accounts() -> None:
    st.title("Topics & Accounts")
    st.caption(
        "Which nutrition subtopics are being hit hardest, and which engagers "
        "show up across multiple coordinated posts (proxy for bot networks)."
    )

    df = analyses_df(limit=2000)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Topic share")
        if df.empty:
            st.info("No data yet.")
        else:
            df_topic = (
                df.assign(topic=df["topic"].fillna("other"))
                .groupby(["topic", "verdict"]).size().reset_index(name="count")
            )
            fig = px.bar(df_topic, x="topic", y="count", color="verdict",
                         color_discrete_map=VERDICT_COLOR, barmode="stack",
                         height=400)
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Coordinated rate by topic")
        if df.empty:
            st.info("No data yet.")
        else:
            df_rate = (
                df.assign(topic=df["topic"].fillna("other"),
                          is_coord=(df["verdict"] == "coordinated").astype(int))
                .groupby("topic").agg(
                    posts=("verdict", "size"),
                    coord_rate=("is_coord", "mean"),
                ).reset_index()
            )
            df_rate = df_rate[df_rate["posts"] >= 2].sort_values("coord_rate", ascending=False)
            if df_rate.empty:
                st.info("Not enough volume per topic yet.")
            else:
                fig = px.bar(df_rate, x="coord_rate", y="topic", orientation="h",
                             color="coord_rate", color_continuous_scale="reds",
                             height=400, hover_data=["posts"])
                fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                  xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Suspected engager network")
    st.caption(
        "Engagers (likers / reposters) ranked by how often they appear on "
        "flagged posts. A handful of IDs appearing on many coordinated posts "
        "is a strong bot-squad signal."
    )

    stats = engager_network_stats()
    if not stats:
        st.info("No engager-level data yet. Run **Daily Check → Run Now** "
                "or analyze a few posts to populate.")
        return

    df_net = pd.DataFrame(stats).head(30)
    df_net["suspicion_pct"] = (df_net["suspicion_score"] * 100).round(0)
    st.dataframe(
        df_net[["engager_id", "n_flagged", "n_total", "suspicion_pct"]],
        use_container_width=True, hide_index=True,
        column_config={
            "engager_id": "Engager",
            "n_flagged": "# flagged posts",
            "n_total": "# total posts",
            "suspicion_pct": st.column_config.NumberColumn(
                "Suspicion", format="%d%%"
            ),
        },
    )

    if len(df_net) > 3:
        fig = px.scatter(df_net, x="n_total", y="suspicion_score",
                         size="n_flagged", color="suspicion_score",
                         color_continuous_scale="reds",
                         hover_name="engager_id", height=400,
                         labels={"n_total": "posts engaged",
                                 "suspicion_score": "share flagged"})
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Section: Targets
# =========================================================

def render_targets() -> None:
    st.title("Monitored Targets")
    st.caption("Hashtags, keywords, and accounts that the daily check sweeps.")

    targets = list_targets()
    if targets:
        df_t = pd.DataFrame(targets)
        df_t["enabled"] = df_t["enabled"].astype(bool)
        # quick stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Targets", len(df_t))
        c2.metric("Enabled", int(df_t["enabled"].sum()))
        c3.metric("Topics covered", df_t["topic"].fillna("other").nunique())

        st.dataframe(
            df_t[["id", "kind", "value", "topic", "enabled", "last_checked_at"]],
            use_container_width=True, hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "kind": "Kind",
                "value": "Value",
                "topic": "Topic",
                "enabled": st.column_config.CheckboxColumn("Enabled"),
                "last_checked_at": st.column_config.DatetimeColumn("Last checked"),
            },
        )

        st.markdown("---")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Toggle / Remove**")
            ids = df_t["id"].tolist()
            chosen = st.selectbox("Target id", ids,
                                  format_func=lambda i: f"#{i}: {df_t.loc[df_t['id']==i,'kind'].iat[0]} "
                                  f"{df_t.loc[df_t['id']==i,'value'].iat[0]}")
            tcol1, tcol2, tcol3 = st.columns(3)
            with tcol1:
                if st.button("Enable", use_container_width=True):
                    set_target_enabled(chosen, True); st.rerun()
            with tcol2:
                if st.button("Disable", use_container_width=True):
                    set_target_enabled(chosen, False); st.rerun()
            with tcol3:
                if st.button("Remove", use_container_width=True, type="secondary"):
                    remove_target(chosen); st.rerun()
    else:
        st.info("No targets yet. Add one below.")

    st.markdown("---")
    st.subheader("Add a target")
    with st.form("add_target_form", clear_on_submit=True):
        a1, a2, a3 = st.columns(3)
        with a1:
            kind = st.selectbox("Kind", ["hashtag", "keyword", "account"])
        with a2:
            value = st.text_input("Value",
                                  placeholder="#keto / detox cleanse / @user")
        with a3:
            topic = st.selectbox("Topic", all_topic_keys(),
                                 index=all_topic_keys().index("other"))
        submitted = st.form_submit_button("Add", type="primary")
        if submitted and value:
            add_target(kind, value.strip(), topic)
            st.success(f"Added {kind}:{value}")
            st.rerun()


# =========================================================
# Section: Reply Queue
# =========================================================

def render_reply_queue() -> None:
    st.title("Reply Queue")
    st.warning(
        "**Human-in-the-loop**. Drafts here have NOT been posted. "
        "Approving a draft only marks it as approved in the local DB — "
        "actual posting to X/Twitter requires you to wire in their API "
        "with your own credentials. Mass-replying may violate platform "
        "terms of service; review carefully and post sparingly."
    )

    tabs = st.tabs(["Pending", "Approved / Sent", "Dismissed"])

    with tabs[0]:
        pending = list_replies(status="pending")
        st.write(f"**{len(pending)}** drafts awaiting review.")
        if not pending:
            st.info("No pending drafts. Run **Daily Check** to populate.")
        for r in pending:
            with st.container(border=True):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**URL**: {r['url']}")
                    st.text_area(
                        "Draft",
                        value=r["draft_text"],
                        key=f"draft_{r['id']}",
                        height=80,
                        label_visibility="collapsed",
                    )
                    if r.get("evidence_json"):
                        with st.expander("Evidence"):
                            try:
                                st.json(json.loads(r["evidence_json"]),
                                        expanded=False)
                            except json.JSONDecodeError:
                                st.write(r["evidence_json"])
                with cols[1]:
                    st.caption(f"Created {r['created_at']}")
                    b1, b2, b3 = st.columns(3)
                    if b1.button("✓ Save edit", key=f"save_{r['id']}",
                                 use_container_width=True):
                        update_reply(
                            r["id"],
                            draft_text=st.session_state[f"draft_{r['id']}"],
                        )
                        st.rerun()
                    if b2.button("Approve", key=f"appr_{r['id']}",
                                 type="primary", use_container_width=True):
                        update_reply(r["id"], status="approved")
                        st.rerun()
                    if b3.button("Dismiss", key=f"dism_{r['id']}",
                                 use_container_width=True):
                        update_reply(r["id"], status="dismissed")
                        st.rerun()

    with tabs[1]:
        approved = list_replies(status="approved") + list_replies(status="sent")
        st.write(f"**{len(approved)}** approved drafts.")
        if approved:
            for r in approved:
                with st.container(border=True):
                    st.markdown(f"**URL**: {r['url']}")
                    st.code(r["draft_text"])
                    cols = st.columns([3, 1])
                    cols[0].caption(f"Status: {r['status']}  "
                                    f"Decided at: {r.get('decided_at')}")
                    if r["status"] == "approved":
                        if cols[1].button("Mark sent", key=f"sent_{r['id']}"):
                            update_reply(r["id"], status="sent")
                            st.rerun()

    with tabs[2]:
        dismissed = list_replies(status="dismissed")
        st.write(f"**{len(dismissed)}** dismissed.")
        if dismissed:
            st.dataframe(
                pd.DataFrame(dismissed)[["url", "draft_text", "decided_at"]],
                use_container_width=True, hide_index=True,
            )


# =========================================================
# Section: Daily Check
# =========================================================

def render_daily_check() -> None:
    st.title("Daily Check")
    st.caption(
        "Sweeps every enabled target, analyzes recent posts, and queues "
        "drafts for coordinated detections."
    )

    targets = list_targets(enabled_only=True)
    st.metric("Enabled targets", len(targets))

    if st.button("Run Now", type="primary"):
        with st.spinner("Running daily check..."):
            from daily_check import run_daily_check
            summary = run_daily_check(verbose=False)
        st.session_state["last_daily_summary"] = summary
        st.success("Done.")

    if "last_daily_summary" in st.session_state:
        s = st.session_state["last_daily_summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Targets checked", s["targets_checked"])
        c2.metric("Posts analyzed", s["posts_analyzed"])
        c3.metric("Posts flagged", s["posts_flagged"])
        c4.metric("Replies drafted", s["replies_drafted"])
        st.caption(f"Started {s['started_at']} · finished {s.get('finished_at')}")
        if s["errors"]:
            st.error("Errors:")
            for e in s["errors"]:
                st.write(f"- {e}")
        with st.expander("Per-target breakdown"):
            st.dataframe(pd.DataFrame(s["per_target"]),
                         use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Backend connection checks")
    st.caption("Safe smoke-tests that confirm the relevant .env credentials work.")

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("Snowflake smoke-check", use_container_width=True):
            with st.spinner("Connecting to Snowflake..."):
                try:
                    import snowflake_client
                    out = snowflake_client.smoke_check()
                except Exception as e:
                    out = {"ok": False, "reason": repr(e)}
            if out.get("ok"):
                st.success(
                    f"Snowflake **{out['snowflake_version']}**  ·  "
                    f"`{out['table']}` has **{out['row_count']:,}** rows"
                )
            else:
                st.error(f"Failed: {out.get('reason')}")

    with cc2:
        if st.button("Scraping Browser smoke-check", use_container_width=True):
            with st.spinner("Connecting to BrightData browser..."):
                try:
                    import scraping_browser_client
                    out = scraping_browser_client.smoke_check()
                except Exception as e:
                    out = {"ok": False, "reason": repr(e)}
            if out.get("ok"):
                st.success(f"Browser connected ✓  ·  test page title: `{out['title']}`")
            else:
                st.error(f"Failed: {out.get('reason')}")

    st.markdown("---")
    st.subheader("BrightData browser sessions")
    st.caption(
        "Live + recent Scraping Browser sessions on your BrightData account, "
        "pulled from `/browser_sessions`. Useful for spotting failures, "
        "tracking spend, and showing the demo that live scraping is happening."
    )
    bs_cols = st.columns([1, 1, 2])
    with bs_cols[0]:
        bs_limit = st.number_input("Limit", min_value=10, max_value=200,
                                    value=50, step=10)
    with bs_cols[1]:
        bs_status = st.selectbox("Status",
                                  ["all", "running", "ended", "errored"], index=0)
    with bs_cols[2]:
        st.write("")
        st.write("")
        bs_refresh = st.button("Refresh sessions", use_container_width=True)

    if bs_refresh:
        with st.spinner("Calling BrightData /browser_sessions..."):
            import brightdata_client
            res = brightdata_client.list_browser_sessions(
                limit=int(bs_limit), status=bs_status
            )
        st.session_state["bd_sessions_result"] = res

    res = st.session_state.get("bd_sessions_result")
    if res is None:
        st.info("Click **Refresh sessions** to fetch from BrightData.")
    elif not res.get("ok"):
        st.error(f"Failed: {res.get('reason')}")
        if res.get("body"):
            with st.expander("Response body"):
                st.code(res["body"])
    else:
        sessions = res["sessions"]
        import brightdata_client
        kpis = brightdata_client.summarize_sessions(sessions)
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Total", kpis["total"])
        kc2.metric("Running", kpis["running"])
        kc3.metric("Errored", kpis["errored"])
        kc4.metric("Avg duration", f"{kpis['avg_duration_s']:.1f}s")

        if sessions:
            df_s = pd.DataFrame(sessions)
            # Try to surface useful columns first if present.
            preferred = [c for c in [
                "id", "session_id", "status", "timestamp", "started_at",
                "duration_s", "duration", "ip", "country", "bytes",
                "target_url", "url",
            ] if c in df_s.columns]
            other = [c for c in df_s.columns if c not in preferred]
            df_view = df_s[preferred + other] if preferred else df_s
            st.dataframe(df_view.head(50), use_container_width=True, hide_index=True)

            # Sessions over time chart (if timestamp-ish column exists)
            ts_col = next((c for c in ("timestamp", "started_at", "created_at")
                           if c in df_s.columns), None)
            if ts_col and len(df_s) >= 5:
                try:
                    df_s["_ts"] = pd.to_datetime(df_s[ts_col], errors="coerce")
                    df_s = df_s.dropna(subset=["_ts"])
                    trend = (
                        df_s.assign(hour=df_s["_ts"].dt.floor("h"))
                        .groupby(["hour", df_s.get("status", "status")
                                  if "status" in df_s.columns else "hour"])
                        .size().reset_index(name="count")
                        if "status" in df_s.columns else
                        df_s.assign(hour=df_s["_ts"].dt.floor("h"))
                        .groupby("hour").size().reset_index(name="count")
                    )
                    fig_s = px.bar(trend, x="hour", y="count",
                                   color="status" if "status" in trend.columns else None,
                                   height=300, title="Sessions per hour")
                    st.plotly_chart(fig_s, use_container_width=True)
                except Exception:
                    pass

    st.markdown("---")
    st.subheader("Schedule daily")
    st.markdown(
        "**Windows Task Scheduler** — set up a 9am daily run from PowerShell:"
    )
    st.code(
        'schtasks /Create /SC DAILY /TN "HawkesDiffusionDaily" /TR '
        '"C:\\Users\\levem\\Projects\\hawkes-diffusion\\.venv\\Scripts\\python.exe '
        'C:\\Users\\levem\\Projects\\hawkes-diffusion\\daily_check.py" /ST 09:00',
        language="powershell",
    )
    st.markdown("**Inspect or remove the schedule:**")
    st.code(
        'schtasks /Query /TN "HawkesDiffusionDaily"\n'
        'schtasks /Delete /TN "HawkesDiffusionDaily" /F',
        language="powershell",
    )


# =========================================================
# Router
# =========================================================
RENDERERS = {
    "Overview": render_overview,
    "Methodology": render_methodology,
    "Analyze Post": render_analyze,
    "Detections": render_detections,
    "Forensics": render_forensics,
    "Topics & Accounts": render_topics_accounts,
    "Targets": render_targets,
    "Reply Queue": render_reply_queue,
    "Daily Check": render_daily_check,
}
RENDERERS[section]()

st.markdown("---")
st.caption(
    "Hackathon build · synthetic Hawkes reference clusters · domain: nutrition · "
    "human-in-the-loop reply queue (no auto-posting)."
)
