"""Vectorized ranking engine shared by the dashboard.

All functions operate on precomputed artifacts (embeddings, sub-score
matrix, penalty multipliers), so re-ranking 100K candidates under new
weights or a new JD embedding is a single matrix multiply.
"""

from typing import Dict, List, Tuple

import numpy as np

from config import SCORE_DECIMALS, SCORE_SCALE

SUBSCORE_ORDER = ["technical_fit", "career_quality", "availability_signal", "seniority_fit"]


def scale_score(raw: float, scale: float = SCORE_SCALE) -> float:
    """Map a raw composite (~[0, 1]) onto a 0..scale display scale.

    Scaling is monotonic, so it preserves ranking order. Raw scores can dip
    slightly below 0 (negative semantic similarity) or brush past 1, so the
    value is clamped to keep the displayed score within [0, scale].
    """
    return max(0.0, min(float(raw), 1.0)) * scale


def format_score(raw: float, scale: float = SCORE_SCALE, decimals: int = SCORE_DECIMALS) -> str:
    """Render a raw composite as a fixed-precision x/scale string."""
    return f"{scale_score(raw, scale):.{decimals}f}"


def build_matrices(candidate_ids, subscores_dict) -> Tuple[np.ndarray, np.ndarray]:
    """Pack the per-candidate subscore dicts into dense arrays once."""
    n = len(candidate_ids)
    subscore_matrix = np.zeros((n, len(SUBSCORE_ORDER)), dtype=np.float32)
    penalties = np.ones(n, dtype=np.float32)
    for i, cid in enumerate(candidate_ids):
        ss = subscores_dict.get(cid, {})
        for j, name in enumerate(SUBSCORE_ORDER):
            subscore_matrix[i, j] = ss.get(name, 0.0)
        penalties[i] = ss.get("penalty_multiplier", 1.0)
    return subscore_matrix, penalties


def compute_scores(
    subscore_matrix: np.ndarray,
    penalties: np.ndarray,
    semantic_sim: np.ndarray,
    weights: Dict[str, float],
) -> np.ndarray:
    w = np.array([weights[name] for name in SUBSCORE_ORDER], dtype=np.float32)
    base = subscore_matrix @ w
    return penalties * (base + weights["semantic_similarity"] * semantic_sim)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores, ordered best-first (deterministic ties)."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.lexsort((idx, -np.round(scores[idx], 6)))]


def mmr_rerank(
    candidate_idx: np.ndarray,
    scores: np.ndarray,
    embeddings: np.ndarray,
    lambda_relevance: float,
    k: int,
) -> List[int]:
    """Maximal Marginal Relevance over a candidate pool: balances score
    against similarity to already-selected profiles so the shortlist
    isn't k near-identical candidates.

    lambda_relevance=1.0 -> pure score ranking; 0.0 -> pure diversity.
    """
    pool = list(candidate_idx)
    if not pool:
        return []
    pool_emb = embeddings[pool]  # normalized embeddings -> dot = cosine
    pool_scores = scores[pool]
    smin, smax = pool_scores.min(), pool_scores.max()
    rel = (pool_scores - smin) / (smax - smin) if smax > smin else np.ones_like(pool_scores)

    selected: List[int] = []
    selected_mask = np.zeros(len(pool), dtype=bool)
    max_sim_to_selected = np.full(len(pool), -1.0, dtype=np.float32)

    for _ in range(min(k, len(pool))):
        if not selected:
            mmr = rel
        else:
            mmr = lambda_relevance * rel - (1.0 - lambda_relevance) * max_sim_to_selected
        mmr = np.where(selected_mask, -np.inf, mmr)
        best = int(np.argmax(mmr))
        selected.append(pool[best])
        selected_mask[best] = True
        sims = pool_emb @ pool_emb[best]
        max_sim_to_selected = np.maximum(max_sim_to_selected, sims)

    return selected


def stability_analysis(
    subscore_matrix: np.ndarray,
    penalties: np.ndarray,
    semantic_sim: np.ndarray,
    weights: Dict[str, float],
    k: int,
    n_trials: int = 200,
    jitter: float = 0.20,
    seed: int = 42,
) -> Dict[int, float]:
    """Perturb each weight by ±jitter (relative, renormalized) n_trials
    times and report, per candidate index, the fraction of trials in
    which it stays in the top-k. Answers \"are the weights arbitrary?\":
    a candidate at 95%+ is robustly ranked, not an artifact of one
    weight choice."""
    rng = np.random.default_rng(seed)
    names = SUBSCORE_ORDER + ["semantic_similarity"]
    base_w = np.array([weights[n] for n in names], dtype=np.float64)

    counts: Dict[int, int] = {}
    for _ in range(n_trials):
        w = base_w * rng.uniform(1.0 - jitter, 1.0 + jitter, size=len(base_w))
        w = w / w.sum()
        scores = penalties * (
            subscore_matrix @ w[:4].astype(np.float32)
            + np.float32(w[4]) * semantic_sim
        )
        for i in np.argpartition(-scores, k - 1)[:k]:
            counts[int(i)] = counts.get(int(i), 0) + 1

    return {i: c / n_trials for i, c in counts.items()}
