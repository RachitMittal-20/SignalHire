"""RecruiterIQ — interactive candidate ranking dashboard.

Re-ranking is live: precomputed artifacts make scoring 100K candidates a
single matrix multiply, so weight sliders and custom job descriptions
re-rank instantly without any pipeline rerun.
"""

import csv
import io
import json
import pickle
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ARTIFACTS_DIR,
    CANDIDATES_PATH,
    EMBEDDING_MODEL,
    SCORE_SCALE,
    TOP_K,
    WEIGHTS,
)
from engine import (
    SUBSCORE_ORDER,
    build_matrices,
    compute_scores,
    format_score,
    mmr_rerank,
    scale_score,
    stability_analysis,
    top_k_indices,
)
from evidence import collect_evidence, generate_reasoning
from rank import build_offset_index, load_candidates_by_ids

st.set_page_config(
    page_title="RecruiterIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#4F8EF7"
GREEN = "#3ECF8E"
AMBER = "#F59E0B"
RED = "#EF4444"
MUTED = "#94A3B8"

SUBSCORE_LABELS = {
    "technical_fit": "Technical fit",
    "career_quality": "Career quality",
    "availability_signal": "Availability",
    "seniority_fit": "Seniority fit",
    "semantic_similarity": "Semantic match",
}

GLOBAL_CSS = """
<style>
  .block-container { padding-top: 2.2rem; }
  div[data-testid="stMetric"] {
    background: #1A1D27;
    border: 1px solid #262b3d;
    border-radius: 10px;
    padding: 10px 14px;
  }
  div[data-testid="stExpander"] details {
    border: 1px solid #262b3d;
    border-radius: 10px;
    background: #161925;
  }
  button[data-baseweb="tab"] { font-size: 0.95rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  h1 { letter-spacing: -0.02em; }
</style>
"""


# ---------------------------------------------------------------- data layer

@st.cache_resource(show_spinner="Loading artifacts ...")
def load_artifacts():
    artifacts = {}
    artifacts["embeddings"] = np.load(str(ARTIFACTS_DIR / "embeddings.npy")).astype(np.float32)
    artifacts["candidate_ids"] = np.load(
        str(ARTIFACTS_DIR / "candidate_ids.npy"), allow_pickle=True
    )
    artifacts["jd_embedding"] = np.load(str(ARTIFACTS_DIR / "jd_embedding.npy")).astype(np.float32)
    with open(ARTIFACTS_DIR / "subscores.pkl", "rb") as f:
        artifacts["subscores"] = pickle.load(f)
    with open(ARTIFACTS_DIR / "disqualified.json", "r") as f:
        artifacts["disqualified"] = json.load(f)
    return artifacts


@st.cache_resource(show_spinner="Packing score matrices ...")
def get_matrices():
    a = load_artifacts()
    subscore_matrix, penalties = build_matrices(a["candidate_ids"], a["subscores"])
    return subscore_matrix, penalties


@st.cache_resource(show_spinner="Loading embedding model (first custom JD only) ...")
def get_model():
    from sentence_transformers import SentenceTransformer

    # CPU is fine for embedding a single query string.
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@st.cache_data(show_spinner=False)
def embed_text(text: str) -> np.ndarray:
    return get_model().encode(text, normalize_embeddings=True).astype(np.float32)


@st.cache_resource(show_spinner="Indexing candidate file (one-time) ...")
def get_offset_index() -> dict:
    return build_offset_index()


@st.cache_data(show_spinner=False)
def cached_candidates(ids: tuple) -> dict:
    # Seek-based loads: any shortlist (every slider move produces a new
    # one) resolves in milliseconds instead of a full 487 MB file scan.
    return load_candidates_by_ids(ids, get_offset_index())


@st.cache_data(show_spinner="Building pool demographics (one-time pass) ...")
def pool_demographics() -> pd.DataFrame:
    """Light per-candidate demographics for the fairness audit. Extracted
    once from the JSONL and cached on disk so later sessions load instantly."""
    cache_path = ARTIFACTS_DIR / "demographics.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = []
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            profile = c.get("profile", {})
            tiers = [
                str(e.get("tier", "unknown")) for e in c.get("education", []) if e.get("tier")
            ]
            best_tier = min(tiers) if tiers else "unknown"
            rows.append(
                {
                    "candidate_id": c.get("candidate_id", ""),
                    "country": profile.get("country", "unknown") or "unknown",
                    "yoe": profile.get("years_of_experience", 0) or 0,
                    "edu_tier": best_tier,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


@st.cache_data(show_spinner=False)
def cached_stability(weights_key: tuple, jd_key: str, k: int) -> dict:
    a = load_artifacts()
    subscore_matrix, penalties = get_matrices()
    weights = dict(weights_key)
    jd_emb = st.session_state.get("jd_embedding_override")
    if jd_emb is None:
        jd_emb = a["jd_embedding"]
    semantic_sim = a["embeddings"] @ jd_emb
    return stability_analysis(subscore_matrix, penalties, semantic_sim, weights, k)


# ---------------------------------------------------------------- ui helpers

def score_bar_html(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(float(value), 1.0)) * 100
    return f"""
    <div style="margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:{MUTED}">
        <span>{label}</span><span>{value:.0%}</span>
      </div>
      <div style="background:#262b3d;border-radius:6px;height:8px">
        <div style="background:{color};width:{pct:.1f}%;height:8px;border-radius:6px"></div>
      </div>
    </div>
    """


def chip(text: str, color: str = ACCENT, title: str = "") -> str:
    return (
        f'<span title="{title}" style="background:{color}22;color:{color};'
        f"border:1px solid {color}55;border-radius:12px;padding:2px 10px;"
        f'margin:2px;font-size:0.75rem;display:inline-block">{text}</span>'
    )


def stability_badge(freq: float) -> str:
    if freq >= 0.90:
        return chip(f"stable {freq:.0%}", GREEN, "Stays in top-100 under ±20% weight perturbation")
    if freq >= 0.60:
        return chip(f"moderate {freq:.0%}", AMBER, "Sensitive to weight choices")
    return chip(f"fragile {freq:.0%}", RED, "Only in top-100 for some weightings")


def normalized_weights() -> dict:
    raw = {name: st.session_state.get(f"w_{name}", WEIGHTS[name]) for name in WEIGHTS}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def _reset_weights():
    # Runs as an on_click callback, *before* the next script run — writing
    # widget keys here is legal, while doing it inline after the sliders
    # are instantiated raises StreamlitAPIException.
    for name in WEIGHTS:
        st.session_state[f"w_{name}"] = float(WEIGHTS[name])


def _load_more(step: int):
    # Reveal `step` more shortlist rows. Runs as an on_click callback so the
    # new count is in place before the shortlist re-renders this run.
    st.session_state["shortlist_n"] = st.session_state.get("shortlist_n", 10) + step


# ------------------------------------------------------------------- sidebar

def render_sidebar(artifacts_ready: bool, n_candidates: int, n_disqualified: int) -> dict:
    with st.sidebar:
        st.markdown("## 🎯 RecruiterIQ")
        st.caption("Live candidate ranking — every control re-ranks instantly.")

        st.markdown("### Job description")
        jd_text = st.text_area(
            "Custom JD or natural-language query",
            height=110,
            placeholder='e.g. "RAG engineer, 5+ yrs, strong vector search, short notice"',
            label_visibility="collapsed",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Apply JD", type="primary", width="stretch", disabled=not jd_text.strip()):
                st.session_state["jd_embedding_override"] = embed_text(jd_text.strip())
                st.session_state["jd_label"] = jd_text.strip()[:60]
        with col_b:
            if st.button("Default JD", width="stretch"):
                st.session_state.pop("jd_embedding_override", None)
                st.session_state.pop("jd_label", None)
        if "jd_label" in st.session_state:
            st.info(f'Ranking against: "{st.session_state["jd_label"]}…"')

        st.markdown("### Signal weights")
        st.caption("Drag to see the shortlist reshuffle live.")
        for name in WEIGHTS:
            st.slider(
                SUBSCORE_LABELS[name],
                min_value=0.0,
                max_value=1.0,
                value=float(WEIGHTS[name]),
                step=0.01,
                key=f"w_{name}",
            )
        weights = normalized_weights()
        st.button("Reset weights", width="stretch", on_click=_reset_weights)
        st.caption(
            "Effective (normalized): "
            + " · ".join(f"{SUBSCORE_LABELS[k].split()[0]} {v:.0%}" for k, v in weights.items())
        )

        st.markdown("### Shortlist")
        diversity = st.slider(
            "Relevance ↔ Diversity (MMR)",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05,
            help="0 = pure score ranking. Higher values penalize near-duplicate "
            "profiles so the shortlist covers distinct candidate archetypes.",
        )
        anonymized = st.toggle(
            "🕶️ Blind screening mode",
            value=False,
            help="Hides names, companies and institutions to reduce reviewer bias.",
        )

        st.divider()
        if artifacts_ready:
            c1, c2 = st.columns(2)
            c1.metric("Candidates", f"{n_candidates:,}")
            c2.metric("Disqualified", n_disqualified)
        else:
            st.warning("Artifacts not loaded — run `python precompute.py` first.")

    return {"weights": weights, "diversity": diversity, "anonymized": anonymized}


# ------------------------------------------------------------------ ranking

def run_ranking(artifacts, weights: dict, diversity: float):
    subscore_matrix, penalties = get_matrices()
    jd_emb = st.session_state.get("jd_embedding_override")
    if jd_emb is None:
        jd_emb = artifacts["jd_embedding"]
    semantic_sim = artifacts["embeddings"] @ jd_emb
    scores = compute_scores(subscore_matrix, penalties, semantic_sim, weights)

    if diversity > 0:
        pool = top_k_indices(scores, TOP_K * 5)
        lambda_rel = 1.0 - diversity
        top_idx = mmr_rerank(pool, scores, artifacts["embeddings"], lambda_rel, TOP_K)
        top_idx = np.array(top_idx)
    else:
        top_idx = top_k_indices(scores, TOP_K)

    return top_idx, scores, semantic_sim, subscore_matrix


# ------------------------------------------------------------ shortlist tab

def candidate_display_name(cand: dict, rank: int, anonymized: bool) -> str:
    profile = cand.get("profile", {})
    if anonymized:
        return f"Candidate #{rank:03d} — {profile.get('current_title', '?')}"
    return (
        f"{profile.get('anonymized_name', '?')} — "
        f"{profile.get('current_title', '?')} @ {profile.get('current_company', '?')}"
    )


def render_evidence(ev: dict):
    chips = []
    for m in ev["matched"]:
        color = GREEN if m["group"] == "must-have" else ACCENT
        label = f"✓ {m['criterion']}"
        if m["source"] == "skill":
            label += f" ← {m['detail']}"
        chips.append(chip(label, color, title=str(m.get("detail", ""))))
    for miss in ev["missing_must_haves"]:
        chips.append(chip(f"✗ {miss['criterion']}", RED, "Missing must-have"))
    if ev["production"]:
        kws = ", ".join(dict.fromkeys(h["keyword"] for h in ev["production"][:3]))
        chips.append(chip(f"🚀 production: {kws}", AMBER))
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    text_hits = [m for m in ev["matched"] if m["source"] == "text" and m["detail"]]
    snippets = [h for h in ev["production"] if h["snippet"]][:1] + [
        {"keyword": m["keyword"], "snippet": m["detail"]} for m in text_hits[:2]
    ]
    if snippets:
        with st.container():
            for s in snippets[:3]:
                st.markdown(
                    f'<div style="border-left:3px solid {ACCENT};padding:4px 10px;'
                    f'margin:4px 0;color:{MUTED};font-size:0.8rem">'
                    f'<b style="color:{ACCENT}">{s["keyword"]}</b>: "{s["snippet"]}"</div>',
                    unsafe_allow_html=True,
                )


def render_shortlist(artifacts, top_idx, scores, semantic_sim, controls, stability):
    ids = artifacts["candidate_ids"]
    subs = artifacts["subscores"]
    top_ids = [str(ids[i]) for i in top_idx]
    candidates_by_id = cached_candidates(tuple(top_ids))

    total = len(top_idx)
    page = 10
    if "shortlist_n" not in st.session_state:
        st.session_state["shortlist_n"] = page
    show_n = min(st.session_state["shortlist_n"], total)

    header_l, header_r = st.columns([3, 1])
    with header_l:
        st.subheader(f"Top {total} candidates")
    with header_r:
        st.caption(f"Showing {show_n} of {total}")

    for rank_pos, i in enumerate(top_idx[:show_n]):
        rank = rank_pos + 1
        cid = str(ids[i])
        cand = candidates_by_id.get(cid)
        if cand is None:
            continue
        ss = subs.get(cid, {})
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        ev = collect_evidence(cand)

        title_line = candidate_display_name(cand, rank, controls["anonymized"])
        with st.expander(
            f"#{rank}  ·  {title_line}  ·  score {format_score(scores[i])}/{SCORE_SCALE:g}",
            expanded=(rank <= 3),
        ):
            col1, col2 = st.columns([1.1, 1.6])
            with col1:
                badge_bits = [stability_badge(stability.get(int(i), 0.0))]
                penalty = ss.get("penalty_multiplier", 1.0)
                if penalty < 1.0:
                    badge_bits.append(chip(f"⚠ penalty ×{penalty:.2f}", RED))
                if signals.get("open_to_work_flag"):
                    badge_bits.append(chip("open to work", GREEN))
                st.markdown(" ".join(badge_bits), unsafe_allow_html=True)

                st.markdown(
                    f"**{profile.get('years_of_experience', '?')} yrs** · "
                    f"{'📍 ' + str(profile.get('location', '?')) if not controls['anonymized'] else '📍 hidden'}"
                )
                if not controls["anonymized"]:
                    st.caption(profile.get("headline", "")[:120])
                rr = signals.get("recruiter_response_rate", 0) or 0
                notice = signals.get("notice_period_days", 90) or 90
                st.markdown(
                    f"Response rate **{rr:.0%}** · notice **{notice}d** · "
                    f"interviews **{(signals.get('interview_completion_rate', 0) or 0):.0%}**"
                )

            with col2:
                bars = [
                    ("Technical fit", ss.get("technical_fit", 0), ACCENT),
                    ("Career quality", ss.get("career_quality", 0), GREEN),
                    ("Availability", ss.get("availability_signal", 0), AMBER),
                    ("Seniority fit", ss.get("seniority_fit", 0), RED),
                    ("Semantic match", float(semantic_sim[i]), MUTED),
                ]
                st.markdown(
                    "".join(score_bar_html(l, v, c) for l, v, c in bars),
                    unsafe_allow_html=True,
                )

            st.markdown("**Why this candidate** — evidence from the profile:")
            render_evidence(ev)
            st.caption(f"*{generate_reasoning(cand, ev)}*")

    if show_n < total:
        remaining = total - show_n
        step = min(page, remaining)
        st.button(
            f"➕ Load {step} more  ·  {remaining} of {total} hidden",
            width="stretch",
            on_click=_load_more,
            args=(step,),
        )
    elif total > page:
        st.caption(f"All {total} candidates shown.")


# -------------------------------------------------------------- insights tab

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,39,0.6)",
    font=dict(color=MUTED),
    margin=dict(l=40, r=20, t=40, b=40),
)


def render_insights(artifacts, top_idx, scores):
    ids = artifacts["candidate_ids"]

    st.subheader("Score landscape")
    pool_n = min(5000, len(scores))
    top_pool = np.array([scale_score(s) for s in np.sort(scores)[-pool_n:]])
    cutoff = scale_score(scores[top_idx].min())
    fig = go.Figure(go.Histogram(x=top_pool, nbinsx=60, marker_color=ACCENT, opacity=0.85))
    fig.add_vline(x=float(cutoff), line_dash="dash", line_color=RED,
                  annotation_text="top-100 cutoff", annotation_font_color=RED)
    fig.update_layout(
        height=300, xaxis_title=f"Composite score (x/{SCORE_SCALE:g})",
        yaxis_title=f"Count (top {pool_n:,})",
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Fairness audit — shortlist vs candidate pool")
    st.caption(
        "If the shortlist's distribution diverges wildly from the qualified pool, "
        "the scoring may be encoding bias. Blind screening mode hides identifying "
        "details during review."
    )
    demo = pool_demographics()
    top_ids = {str(ids[i]) for i in top_idx}
    demo_top = demo[demo["candidate_id"].isin(top_ids)]

    col1, col2 = st.columns(2)
    with col1:
        pool_tier = demo["edu_tier"].value_counts(normalize=True).sort_index()
        top_tier = demo_top["edu_tier"].value_counts(normalize=True).sort_index()
        fig = go.Figure()
        fig.add_bar(name="Pool (100K)", x=pool_tier.index, y=pool_tier.values, marker_color=MUTED)
        fig.add_bar(name="Shortlist", x=top_tier.index, y=top_tier.values, marker_color=ACCENT)
        fig.update_layout(title="Education tier", barmode="group", height=300,
                          yaxis_tickformat=".0%", **PLOT_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with col2:
        top_countries = demo["country"].value_counts().head(8).index
        pool_c = (
            demo[demo["country"].isin(top_countries)]["country"].value_counts(normalize=True)
        )
        top_c = (
            demo_top[demo_top["country"].isin(top_countries)]["country"]
            .value_counts(normalize=True)
            .reindex(pool_c.index)
            .fillna(0)
        )
        fig = go.Figure()
        fig.add_bar(name="Pool (100K)", x=pool_c.index, y=pool_c.values, marker_color=MUTED)
        fig.add_bar(name="Shortlist", x=top_c.index, y=top_c.values, marker_color=GREEN)
        fig.update_layout(title="Country (top 8)", barmode="group", height=300,
                          yaxis_tickformat=".0%", **PLOT_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_histogram(x=demo["yoe"], histnorm="probability", nbinsx=40,
                      name="Pool (100K)", marker_color=MUTED, opacity=0.6)
    fig.add_histogram(x=demo_top["yoe"], histnorm="probability", nbinsx=40,
                      name="Shortlist", marker_color=AMBER, opacity=0.7)
    fig.update_layout(title="Years of experience", barmode="overlay", height=300,
                      yaxis_tickformat=".0%", **PLOT_LAYOUT)
    st.plotly_chart(fig, width="stretch")


# -------------------------------------------------------------- integrity tab

def render_integrity(artifacts):
    disqualified = artifacts.get("disqualified", [])
    honeypots = [d for d in disqualified if d.get("type") == "HONEYPOT"]
    ghosts = [d for d in disqualified if d.get("type") == "GHOST"]
    research = [d for d in disqualified if d.get("type") == "PURE_RESEARCH"]

    st.subheader("Adversarial profile detection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disqualified", len(disqualified))
    c2.metric("🍯 Honeypots", len(honeypots), help="Claimed experience impossible vs career timeline")
    c3.metric("👻 Ghosts", len(ghosts), help="Near-empty profile, nothing verified")
    c4.metric("🔬 Pure research", len(research), help="All research titles, zero deployment evidence")

    st.markdown(
        "The dataset seeds fake profiles to catch rankers that trust self-reported "
        "numbers. These never reach scoring — they are disqualified during ingest."
    )

    if honeypots:
        st.markdown("#### Caught in the act")
        example_ids = tuple(d["id"] for d in honeypots[:3])
        examples = cached_candidates(example_ids)
        for d in honeypots[:3]:
            cand = examples.get(d["id"])
            if not cand:
                continue
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            starts = [r.get("start_date", "")[:4] for r in career if r.get("start_date")]
            earliest = min(starts) if starts else "?"
            st.markdown(
                f"> **{d['id']}** claims **{profile.get('years_of_experience', '?')} years** "
                f"of experience, but their earliest role starts in **{earliest}**. "
                f"_{d.get('reason', '')}_"
            )

    if disqualified:
        with st.expander("Full disqualification log"):
            st.dataframe(pd.DataFrame(disqualified), width="stretch", hide_index=True)


# --------------------------------------------------------------- compare tab

def render_compare(artifacts, top_idx, scores, semantic_sim, anonymized: bool):
    ids = artifacts["candidate_ids"]
    subs = artifacts["subscores"]
    top_ids = [str(ids[i]) for i in top_idx]
    candidates_by_id = cached_candidates(tuple(top_ids))

    def fmt(cid):
        cand = candidates_by_id.get(cid, {})
        rank = top_ids.index(cid) + 1
        return candidate_display_name(cand, rank, anonymized) if cand else cid

    st.subheader("Side-by-side comparison")
    selected = st.multiselect(
        "Pick 2–4 candidates from the shortlist",
        options=top_ids,
        default=top_ids[:3],
        max_selections=4,
        format_func=fmt,
    )
    if len(selected) < 2:
        st.info("Select at least two candidates to compare.")
        return

    axes = list(SUBSCORE_LABELS.values())
    palette = [ACCENT, GREEN, AMBER, RED]
    fig = go.Figure()
    for color, cid in zip(palette, selected):
        i = top_ids.index(cid)
        gi = top_idx[i]
        ss = subs.get(cid, {})
        values = [ss.get(k, 0.0) for k in SUBSCORE_ORDER] + [float(semantic_sim[gi])]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=axes + axes[:1],
                name=f"#{i + 1} " + (cid if anonymized else
                      candidates_by_id.get(cid, {}).get("profile", {}).get("anonymized_name", cid)),
                line=dict(color=color),
                fill="toself",
                opacity=0.55,
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(26,29,39,0.6)",
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#2D3148"),
            angularaxis=dict(gridcolor="#2D3148"),
        ),
        height=460,
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")

    cols = st.columns(len(selected))
    for col, cid in zip(cols, selected):
        cand = candidates_by_id.get(cid)
        if not cand:
            continue
        i = top_ids.index(cid)
        with col:
            st.markdown(f"**#{i + 1} · score {format_score(scores[top_idx[i]])}/{SCORE_SCALE:g}**")
            ev = collect_evidence(cand)
            st.caption(generate_reasoning(cand, ev))
            missing = ", ".join(m["criterion"] for m in ev["missing_must_haves"]) or "none"
            st.markdown(f"Missing must-haves: *{missing}*")


# ---------------------------------------------------------------- export tab

def build_submission_csv(top_idx, scores, ids, candidates_by_id) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    # Sort on the emitted (rescaled, rounded) score so the CSV is provably
    # non-increasing at output precision, with candidate_id ascending as the
    # tie-break the validator requires for equal scores.
    rows = sorted(
        ((str(ids[i]), float(scores[i])) for i in top_idx),
        key=lambda r: (-float(format_score(r[1])), r[0]),
    )
    for rank_pos, (cid, raw) in enumerate(rows):
        cand = candidates_by_id.get(cid)
        writer.writerow(
            [cid, rank_pos + 1, format_score(raw), generate_reasoning(cand or {})]
        )
    return buf.getvalue()


def build_outreach_pack(top_idx, scores, ids, candidates_by_id, n: int = 10) -> str:
    lines = ["# Recruiter Outreach Pack", ""]
    for rank_pos, i in enumerate(top_idx[:n]):
        cid = str(ids[i])
        cand = candidates_by_id.get(cid)
        if not cand:
            continue
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        ev = collect_evidence(cand)
        hooks = [m["detail"] for m in ev["matched"] if m["source"] == "skill"][:3]
        hook_str = ", ".join(dict.fromkeys(hooks)) or "your ML background"
        name = profile.get("anonymized_name", cid)
        notice = signals.get("notice_period_days", 90) or 90
        lines += [
            f"## #{rank_pos + 1} — {name} ({cid}) · score {format_score(scores[i])}/{SCORE_SCALE:g}",
            "",
            f"*{generate_reasoning(cand, ev)}*",
            "",
            "**Draft outreach:**",
            "",
            f"> Hi {name.split()[0] if name else 'there'}, your experience with "
            f"{hook_str} at {profile.get('current_company', 'your current company')} "
            f"stood out for a Senior AI Engineer role we're hiring for — the team "
            f"builds production retrieval and ranking systems. Given your "
            f"{notice}-day notice period, the timing could work well. "
            f"Open to a quick chat this week?",
            "",
        ]
    return "\n".join(lines)


def render_export(artifacts, top_idx, scores, weights):
    ids = artifacts["candidate_ids"]
    top_ids = [str(ids[i]) for i in top_idx]
    candidates_by_id = cached_candidates(tuple(top_ids))

    st.subheader("Take the shortlist with you")
    st.caption("Exports reflect the current weights, JD and diversity settings.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 Submission CSV",
            data=build_submission_csv(top_idx, scores, ids, candidates_by_id),
            file_name="submission.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(f"Top-100 CSV (id, rank, score out of {SCORE_SCALE:g}, evidence-based reasoning).")
    with col2:
        st.download_button(
            "✉️ Outreach pack (top 10)",
            data=build_outreach_pack(top_idx, scores, ids, candidates_by_id),
            file_name="outreach_pack.md",
            mime="text/markdown",
            width="stretch",
        )
        st.caption("Personalized first-touch drafts citing each candidate's actual skills.")
    with col3:
        config_json = json.dumps(
            {
                "weights": weights,
                "jd": st.session_state.get("jd_label", "default"),
                "shortlist": top_ids,
            },
            indent=2,
        )
        st.download_button(
            "⚙️ Ranking config JSON",
            data=config_json,
            file_name="ranking_config.json",
            mime="application/json",
            width="stretch",
        )
        st.caption("Reproducible snapshot: weights, JD and resulting shortlist.")


# --------------------------------------------------------------- main layout

def main():
    try:
        artifacts = load_artifacts()
        artifacts_ready = True
    except Exception as e:
        artifacts = None
        artifacts_ready = False
        st.error(f"Artifacts not found ({e}). Run `python precompute.py` first.")

    n = len(artifacts["candidate_ids"]) if artifacts_ready else 0
    n_disq = len(artifacts["disqualified"]) if artifacts_ready else 0
    controls = render_sidebar(artifacts_ready, n, n_disq)

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.title("🎯 RecruiterIQ")
    st.caption(
        "Explainable AI candidate ranking · 100K profiles re-ranked live · Redrob AI Challenge"
    )

    if not artifacts_ready:
        st.stop()

    t0 = time.perf_counter()
    top_idx, scores, semantic_sim, subscore_matrix = run_ranking(
        artifacts, controls["weights"], controls["diversity"]
    )
    rank_ms = (time.perf_counter() - t0) * 1000
    weights_key = tuple(sorted(controls["weights"].items()))
    jd_key = st.session_state.get("jd_label", "__default__")
    stability = cached_stability(weights_key, jd_key, TOP_K)

    avg_stability = float(np.mean([stability.get(int(i), 0.0) for i in top_idx]))
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pool ranked", f"{n:,}", help="Candidates scored on this rerun")
    k2.metric("Re-rank time", f"{rank_ms:.0f} ms", help="Full 100K re-score on this interaction")
    k3.metric("Top score", f"{format_score(scores[top_idx[0]])}/{SCORE_SCALE:g}")
    k4.metric("Top-100 cutoff", f"{format_score(scores[top_idx].min())}/{SCORE_SCALE:g}")
    k5.metric(
        "Shortlist stability",
        f"{avg_stability:.0%}",
        help="Average fraction of ±20% weight perturbations in which shortlist members stay top-100",
    )

    tabs = st.tabs(
        ["🏆 Shortlist", "⚖️ Compare", "📊 Insights", "🛡️ Integrity", "📤 Export", "📖 Methodology"]
    )

    with tabs[0]:
        render_shortlist(artifacts, top_idx, scores, semantic_sim, controls, stability)
    with tabs[1]:
        render_compare(artifacts, top_idx, scores, semantic_sim, controls["anonymized"])
    with tabs[2]:
        render_insights(artifacts, top_idx, scores)
    with tabs[3]:
        render_integrity(artifacts)
    with tabs[4]:
        render_export(artifacts, top_idx, scores, controls["weights"])
    with tabs[5]:
        render_methodology(controls["weights"])


def render_methodology(weights: dict):
    st.subheader("Composite score")
    formula = " + ".join(
        f"{weights[name]:.2f} × {SUBSCORE_LABELS[name].replace(' ', '_')}" for name in weights
    )
    st.code(f"S = penalty × ({formula})")
    st.markdown(
        """
        | Signal | What it measures |
        |---|---|
        | Technical fit | JD must-haves (embeddings/retrieval, vector DBs, Python, eval) + nice-to-haves, weighted by declared proficiency and assessment scores |
        | Career quality | Product-company history, median tenure, upward title progression |
        | Availability | Open-to-work, recency, response rate, interview completion, notice period |
        | Seniority fit | Ideal 6–9 yrs experience band, education tier bonus |
        | Semantic match | MiniLM embedding cosine vs the JD — catches strong profiles that use plain language |
        """
    )
    st.subheader("Integrity rules")
    from config import (
        CONSULTING_PENALTY,
        CV_SPEECH_ROBOTICS_PENALTY,
        GHOST_COMPLETENESS_THRESHOLD,
        HONEYPOT_YEAR_BUFFER,
        NO_CODE_PENALTY,
    )

    st.markdown(
        f"""
        - **Honeypot** (disqualified): claimed experience exceeds the career
          timeline by more than **{HONEYPOT_YEAR_BUFFER} years**.
        - **Ghost** (disqualified): profile completeness below
          **{GHOST_COMPLETENESS_THRESHOLD}** with no verified email or phone.
        - **Pure research** (disqualified): all roles are research titles with
          zero production/deployment evidence.
        - **All-consulting career**: composite ×**{CONSULTING_PENALTY}**.
        - **No code shipped in 18 months**: composite ×**{NO_CODE_PENALTY}**.
        - **CV/speech/robotics-only ML profile**: composite ×**{CV_SPEECH_ROBOTICS_PENALTY}**.
        """
    )


if __name__ == "__main__":
    main()
