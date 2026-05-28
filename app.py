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


# ---------- Sidebar nav ----------
SECTIONS = [
    "Overview",
    "Analyze Post",
    "Detections",
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
    st.title("Overview")
    st.caption(
        "Last 7 days of detections across your monitored nutrition targets, "
        "with the most recent flags surfaced for quick triage."
    )

    df = analyses_df(limit=2000)
    if df.empty:
        st.info("No analyses yet. Try **Analyze Post** with any URL, or "
                "run **Daily Check → Run Now** to populate the dashboard.")
        return

    now = pd.Timestamp.utcnow().tz_localize(None)
    last_7 = df[df["analyzed_at"] >= now - pd.Timedelta(days=7)]
    last_24 = df[df["analyzed_at"] >= now - pd.Timedelta(hours=24)]

    # KPI row
    n_total_7d = len(last_7)
    n_flag_7d = int((last_7["verdict"] == "coordinated").sum())
    n_flag_24h = int((last_24["verdict"] == "coordinated").sum())
    pending = len(list_replies(status="pending"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Analyzed (7d)", n_total_7d)
    k2.metric("Flagged (7d)", n_flag_7d,
              f"{(n_flag_7d/max(n_total_7d,1))*100:.0f}% rate")
    k3.metric("Flagged (24h)", n_flag_24h)
    k4.metric("Pending replies", pending)

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

    sub_map, sub_feat, sub_rel = st.tabs(
        ["Fingerprint Map", "Feature Breakdown", "Related Posts"]
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
    "Analyze Post": render_analyze,
    "Detections": render_detections,
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
