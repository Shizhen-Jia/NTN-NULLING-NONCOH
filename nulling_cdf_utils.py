from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from BeamformingCalc import nulling_bf, nulling_bf_music_noncoh, svd_bf
from ntn_music_detection import (
    build_ntn_truth_from_paths,
    collapse_cir_to_narrowband,
    run_music_standard_pipeline,
    summarize_ntn_music_quality,
)


def _safe_db(power_linear: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    arr = np.asarray(power_linear, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(arr, float(eps)))
    if np.isscalar(power_linear):
        return float(out)
    return out


def _interference_power_per_rx(h_ntn_tx: np.ndarray, beam: np.ndarray) -> np.ndarray:
    """Per-NTN received interference power for one TX beam."""
    h = np.asarray(h_ntn_tx, dtype=np.complex128)
    v = np.asarray(beam, dtype=np.complex128).reshape(-1)
    if h.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")
    if h.shape[2] != v.shape[0]:
        raise ValueError(
            f"Beam dimension mismatch: h_ntn_tx has {h.shape[2]} TX antennas, beam has {v.shape[0]}."
        )
    # Match the legacy Lambda_CDF_det notebook metric:
    #   |w_t^H h_i|^2  (or |v_null^H h_i|^2),
    # where each per-user channel vector is treated as a TX-antenna column vector.
    eff = np.einsum("nra,a->nr", h, np.conjugate(v), optimize=True)
    return np.sum(np.abs(eff) ** 2, axis=1).real.astype(np.float64)


def _tn_link_power(
    h_tn_link: np.ndarray,
    beam: np.ndarray,
    rx_combiner: np.ndarray,
) -> float:
    """Received TN-link power |v^H H w_r|^2 for one BS-sector beam."""
    h = np.asarray(h_tn_link, dtype=np.complex128)
    v = np.asarray(beam, dtype=np.complex128)
    w_r = np.asarray(rx_combiner, dtype=np.complex128)

    if h.ndim != 2:
        raise ValueError("h_tn_link must have shape (num_tx_ant, num_tn_rx_ant).")
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if w_r.ndim == 1:
        w_r = w_r.reshape(-1, 1)
    if v.ndim != 2 or v.shape[1] != 1:
        raise ValueError(f"beam must have shape (num_tx_ant, 1); got {v.shape}.")
    if w_r.ndim != 2 or w_r.shape[1] != 1:
        raise ValueError(f"rx_combiner must have shape (num_tn_rx_ant, 1); got {w_r.shape}.")
    if h.shape[0] != v.shape[0]:
        raise ValueError(
            f"Beam dimension mismatch: h_tn_link has {h.shape[0]} TX antennas, beam has {v.shape[0]}."
        )
    if h.shape[1] != w_r.shape[0]:
        raise ValueError(
            "RX combiner dimension mismatch: "
            f"h_tn_link has {h.shape[1]} TN RX antennas, rx_combiner has {w_r.shape[0]}."
        )
    return float(np.abs((v.conj().T @ h @ w_r).item()) ** 2)


def _covariance_from_channel_vectors(
    h_vectors: np.ndarray,
    *,
    num_tx_ant: int,
) -> np.ndarray:
    """Build sum_k h_k h_k^H from channel vectors."""
    h = np.asarray(h_vectors, dtype=np.complex128)
    if h.size == 0:
        return np.zeros((num_tx_ant, num_tx_ant), dtype=np.complex128)
    if h.ndim == 1:
        h = h.reshape(1, -1)
    if h.ndim != 2 or h.shape[1] != num_tx_ant:
        raise ValueError(f"h_vectors must have shape (K, {num_tx_ant}); got {h.shape}.")
    return np.einsum("ka,kb->ab", h, np.conjugate(h), optimize=True)


def _channel_vectors_to_noncoh_terms(
    h_vectors: np.ndarray,
    *,
    num_tx_ant: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert channel vectors h_k into normalized directions u_k and powers g_k.

    This makes
        sum_k h_k h_k^H = sum_k g_k u_k u_k^H
    with g_k = ||h_k||^2 and ||u_k|| = 1.
    """
    h = np.asarray(h_vectors, dtype=np.complex128)
    if h.size == 0:
        return (
            np.empty((0, num_tx_ant), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
        )
    if h.ndim == 1:
        h = h.reshape(1, -1)
    if h.ndim != 2 or h.shape[1] != num_tx_ant:
        raise ValueError(f"h_vectors must have shape (K, {num_tx_ant}); got {h.shape}.")

    finite_rows = np.all(np.isfinite(np.real(h)) & np.isfinite(np.imag(h)), axis=1)
    h = h[finite_rows]
    if h.size == 0:
        return (
            np.empty((0, num_tx_ant), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
        )

    norms = np.linalg.norm(h, axis=1)
    valid = np.isfinite(norms) & (norms > float(eps))
    h = h[valid]
    norms = norms[valid]
    if h.size == 0:
        return (
            np.empty((0, num_tx_ant), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
        )

    u = h / norms[:, None]
    g = np.square(norms).astype(np.float64)
    return np.asarray(u, dtype=np.complex128), np.asarray(g, dtype=np.float64)


def _rowwise_vector_correlation(
    ref_vectors: np.ndarray,
    est_vectors: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-row normalized correlation rho = |u_ref^H u_est| / (||u_ref|| ||u_est||)."""
    ref = np.asarray(ref_vectors, dtype=np.complex128)
    est = np.asarray(est_vectors, dtype=np.complex128)

    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if est.ndim == 1:
        est = est.reshape(1, -1)
    if ref.shape != est.shape:
        raise ValueError(
            f"ref_vectors and est_vectors must have the same shape; got {ref.shape} and {est.shape}."
        )
    if ref.size == 0:
        return np.empty((0,), dtype=np.float64)

    ref_norm = np.linalg.norm(ref, axis=1)
    est_norm = np.linalg.norm(est, axis=1)
    denom = ref_norm * est_norm

    rho = np.zeros((ref.shape[0],), dtype=np.float64)
    valid = np.isfinite(ref_norm) & np.isfinite(est_norm) & (denom > float(eps))
    if np.any(valid):
        inner = np.sum(np.conjugate(ref[valid]) * est[valid], axis=1)
        rho[valid] = np.abs(inner) / np.maximum(denom[valid], float(eps))
    return np.clip(rho, 0.0, 1.0)


def summarize_music_noncoh_quality(
    h_ntn_all: np.ndarray,
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    *,
    max_detected_b_terms: str | int | None = "all",
    eps: float = 1e-12,
) -> Dict[str, float | int]:
    """Summarize per-simulation MUSIC noncoherent `(u, g)` quality."""
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx, num_tx_ant).")

    num_tx_total = int(h_ntn.shape[2])
    num_tx_ant = int(h_ntn.shape[3])
    u_rho_all: List[np.ndarray] = []
    g_rel_err_all: List[np.ndarray] = []
    tx_with_pairs = 0

    for tx_idx in range(num_tx_total):
        lookup = music_lookup.get(int(tx_idx), {})
        tx_inputs = _extract_tx_detected_pairs(
            h_ntn[:, :, tx_idx, :],
            lookup,
            num_tx_ant=num_tx_ant,
            max_detected_b_terms=max_detected_b_terms,
        )
        h_true_t = np.asarray(tx_inputs["h_true"], dtype=np.complex128)
        if h_true_t.size == 0:
            continue

        u_est_t = np.asarray(tx_inputs["u"], dtype=np.complex128)
        g_est_t = np.asarray(tx_inputs["g"], dtype=np.float64).reshape(-1)
        u_true_t, g_true_t = _channel_vectors_to_noncoh_terms(
            h_true_t,
            num_tx_ant=num_tx_ant,
            eps=eps,
        )
        if u_true_t.shape != u_est_t.shape or g_true_t.shape != g_est_t.shape:
            raise ValueError(
                "Inconsistent MUSIC noncoherent summary shapes: "
                f"u_true={u_true_t.shape}, u_est={u_est_t.shape}, "
                f"g_true={g_true_t.shape}, g_est={g_est_t.shape}."
            )
        if u_true_t.shape[0] == 0:
            continue

        u_rho_t = _rowwise_vector_correlation(
            u_true_t,
            u_est_t,
            eps=eps,
        )
        g_rel_err_t = np.abs(g_true_t - g_est_t) / np.maximum(np.abs(g_true_t), float(eps))
        valid = np.isfinite(u_rho_t) & np.isfinite(g_rel_err_t)
        if not np.any(valid):
            continue

        tx_with_pairs += 1
        u_rho_all.append(np.asarray(u_rho_t[valid], dtype=np.float64))
        g_rel_err_all.append(np.asarray(g_rel_err_t[valid], dtype=np.float64))

    if len(u_rho_all) == 0:
        return {
            "pairs": 0,
            "tx_with_pairs": 0,
            "u_rho_mean": float("nan"),
            "u_err_mean": float("nan"),
            "g_rel_err_mean": float("nan"),
        }

    u_rho_arr = np.concatenate(u_rho_all, axis=0)
    g_rel_err_arr = np.concatenate(g_rel_err_all, axis=0)
    return {
        "pairs": int(u_rho_arr.size),
        "tx_with_pairs": int(tx_with_pairs),
        "u_rho_mean": float(np.mean(u_rho_arr)),
        "u_err_mean": float(np.mean(1.0 - u_rho_arr)),
        "g_rel_err_mean": float(np.mean(g_rel_err_arr)),
    }


def _mask_channel_tensor_to_pairs(
    h_ntn_tx: np.ndarray,
    pair_rx: np.ndarray,
    pair_rx_ant: np.ndarray,
) -> np.ndarray:
    """Keep only selected `(rx, rx_ant)` channel vectors and zero the rest."""
    h = np.asarray(h_ntn_tx, dtype=np.complex128)
    if h.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")

    pair_rx_arr = np.asarray(pair_rx, dtype=int).reshape(-1)
    pair_rx_ant_arr = np.asarray(pair_rx_ant, dtype=int).reshape(-1)
    if pair_rx_arr.size != pair_rx_ant_arr.size:
        raise ValueError(
            f"pair_rx and pair_rx_ant must have the same length; got "
            f"{pair_rx_arr.size} and {pair_rx_ant_arr.size}."
        )

    masked = np.zeros_like(h)
    if pair_rx_arr.size == 0:
        return masked

    valid = (
        (pair_rx_arr >= 0)
        & (pair_rx_arr < h.shape[0])
        & (pair_rx_ant_arr >= 0)
        & (pair_rx_ant_arr < h.shape[1])
    )
    if not np.any(valid):
        return masked

    masked[pair_rx_arr[valid], pair_rx_ant_arr[valid], :] = h[
        pair_rx_arr[valid],
        pair_rx_ant_arr[valid],
        :,
    ]
    return masked


def _resolve_b_term_limit(max_detected_b_terms: str | int | None) -> int | None:
    """Normalize the user-facing detected-pair limit."""
    if max_detected_b_terms is None:
        return None

    if isinstance(max_detected_b_terms, str):
        value = max_detected_b_terms.strip().lower()
        if value == "all":
            return None
        if value.isdigit():
            max_detected_b_terms = int(value)
        else:
            raise ValueError(
                "max_detected_b_terms must be 'all', None, or a positive integer; "
                f"got {max_detected_b_terms!r}."
            )

    if isinstance(max_detected_b_terms, (int, np.integer)):
        limit = int(max_detected_b_terms)
        if limit <= 0:
            raise ValueError(
                f"max_detected_b_terms must be positive when numeric; got {limit}."
            )
        return limit

    raise ValueError(
        "max_detected_b_terms must be 'all', None, or a positive integer; "
        f"got type {type(max_detected_b_terms).__name__}."
    )


def _pair_rank_score(selection_score: np.ndarray, gain: np.ndarray) -> np.ndarray:
    score = np.asarray(selection_score, dtype=float).reshape(-1)
    gain_arr = np.asarray(gain, dtype=np.float64).reshape(-1)
    if score.size != gain_arr.size:
        raise ValueError(
            f"selection_score and gain must have the same length; got {score.size} and {gain_arr.size}."
        )
    rank_score = score.copy()
    invalid = ~np.isfinite(rank_score)
    if np.any(invalid):
        rank_score[invalid] = gain_arr[invalid]
    return rank_score


def _dedupe_tx_pair_candidates(
    pair_rx: np.ndarray,
    pair_rx_ant: np.ndarray,
    selection_score: np.ndarray,
    u: np.ndarray,
    g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep at most one candidate for each `(rx, rx_ant)` pair within one TX."""
    pair_rx_arr = np.asarray(pair_rx, dtype=int).reshape(-1)
    pair_rx_ant_arr = np.asarray(pair_rx_ant, dtype=int).reshape(-1)
    selection_score_arr = np.asarray(selection_score, dtype=float).reshape(-1)
    u_arr = np.asarray(u, dtype=np.complex128)
    g_arr = np.asarray(g, dtype=np.float64).reshape(-1)

    if pair_rx_arr.size <= 1:
        return pair_rx_arr, pair_rx_ant_arr, selection_score_arr, u_arr, g_arr

    rank_score = _pair_rank_score(selection_score_arr, g_arr)
    best_idx_by_key: Dict[Tuple[int, int], int] = {}

    for idx, (rx_i, ant_i) in enumerate(zip(pair_rx_arr.tolist(), pair_rx_ant_arr.tolist())):
        key = (int(rx_i), int(ant_i))
        prev_idx = best_idx_by_key.get(key)
        if prev_idx is None or rank_score[idx] > rank_score[prev_idx]:
            best_idx_by_key[key] = idx

    keep_idx = np.array(sorted(best_idx_by_key.values()), dtype=int)
    return (
        pair_rx_arr[keep_idx],
        pair_rx_ant_arr[keep_idx],
        selection_score_arr[keep_idx],
        u_arr[keep_idx],
        g_arr[keep_idx],
    )


def pair_tn_to_strongest_tx(
    h_tn_all: np.ndarray,
    *,
    h_tn_th: float,
    tx_antennas: int,
    tx_power: float,
    snr_noise_power: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Pair each TN to its strongest valid TX over all BS sectors."""
    h = np.asarray(h_tn_all, dtype=np.complex128)
    if h.ndim != 4:
        raise ValueError("h_tn_all must have shape (num_tn_rx, num_tn_rx_ant, num_tx, num_tx_ant).")

    num_tn_rx, _num_tn_rx_ant, num_tx_total, _num_tx_ant = h.shape
    h_flat = np.transpose(h, (0, 2, 1, 3)).reshape(num_tn_rx, num_tx_total, -1)
    h_norms = np.linalg.norm(h_flat, axis=2)

    nonzero_mask = h_norms > float(eps)
    valid_mask = nonzero_mask & (h_norms > float(h_tn_th))

    best_tx_idx = np.full((num_tn_rx,), -1, dtype=int)
    best_h_norm = np.zeros((num_tn_rx,), dtype=np.float64)
    pairs_by_tx: Dict[int, List[Dict[str, Any]]] = {int(t): [] for t in range(num_tx_total)}

    for tn_idx in range(num_tn_rx):
        tx_candidates = np.flatnonzero(valid_mask[tn_idx])
        if tx_candidates.size == 0:
            continue

        tx_local_best = tx_candidates[int(np.argmax(h_norms[tn_idx, tx_candidates]))]
        best_tx_idx[tn_idx] = int(tx_local_best)
        best_h_norm[tn_idx] = float(h_norms[tn_idx, tx_local_best])

        h_tn = np.asarray(h[tn_idx, :, tx_local_best, :], dtype=np.complex128).T
        w_t, w_r = svd_bf(h_tn, tx_antennas)
        snr_raw_linear = (
            np.abs((w_t.conj().T @ h_tn @ w_r).item()) ** 2
            * float(tx_power)
            / float(snr_noise_power)
        )

        pairs_by_tx[int(tx_local_best)].append(
            {
                "tn_idx": int(tn_idx),
                "tx_idx": int(tx_local_best),
                "h_tn": h_tn,
                "h_norm": float(best_h_norm[tn_idx]),
                "w_t": np.asarray(w_t, dtype=np.complex128),
                "w_r": np.asarray(w_r, dtype=np.complex128),
                "snr_raw_db": float(_safe_db(snr_raw_linear, eps=eps)),
            }
        )

    pair_counts_by_tx = np.array([len(pairs_by_tx[int(t)]) for t in range(num_tx_total)], dtype=int)
    min_count = int(pair_counts_by_tx.min()) if pair_counts_by_tx.size > 0 else 0

    return {
        "h_norms": h_norms,
        "nonzero_mask": nonzero_mask,
        "valid_mask": valid_mask,
        "best_tx_idx": best_tx_idx,
        "best_h_norm": best_h_norm,
        "pairs_by_tx": pairs_by_tx,
        "pair_counts_by_tx": pair_counts_by_tx,
        "min_count": min_count,
    }


def build_music_tx_lookup(
    ntn_music_out: Dict[str, Any],
    *,
    num_ntn_rx: int,
    num_tx_total: int,
    num_tx_ant: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Collect detected NTN pair data per TX for MUSIC-guided nulling."""
    pair_rx = np.asarray(ntn_music_out.get("pair_rx_idx", []), dtype=int)
    pair_t = np.asarray(ntn_music_out.get("pair_t_idx", []), dtype=int)
    # pair_u = np.asarray(ntn_music_out.get("pair_u_hat", []), dtype=np.complex128)
    pair_u = np.asarray(ntn_music_out.get("pair_u_hat_raw", []), dtype=np.complex128)
    pair_alpha_hat = np.asarray(ntn_music_out.get("pair_alpha_hat_raw", []), dtype=np.complex128)
    pair_rx_ant_idx = np.asarray(ntn_music_out.get("pair_rx_ant_idx", []), dtype=int)
    pair_score_user = np.asarray(ntn_music_out.get("pair_score_user", []), dtype=float)
    pair_fit_score = np.asarray(ntn_music_out.get("pair_fit_score", []), dtype=float)
    pair_selection_score = np.asarray(ntn_music_out.get("pair_selection_score", []), dtype=float)

    if pair_u.ndim == 1:
        if pair_u.size == 0:
            pair_u = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif pair_u.size == num_tx_ant:
            pair_u = pair_u.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected pair_u shape: {pair_u.shape}")
    if pair_u.ndim != 2:
        raise ValueError(f"pair_u_hat must be 2D after reshape, got {pair_u.shape}")
    if pair_u.shape[1] != num_tx_ant and pair_u.shape[0] > 0:
        raise ValueError(
            f"pair_u_hat antenna dimension mismatch: expected {num_tx_ant}, got {pair_u.shape[1]}."
        )

    if not (
        pair_rx.size
        == pair_t.size
        == pair_rx_ant_idx.size
        == pair_alpha_hat.size
        == pair_u.shape[0]
    ):
        raise ValueError("Inconsistent MUSIC pair lengths in ntn_music_out.")
    if pair_score_user.size not in (0, pair_rx.size):
        raise ValueError("pair_score_user length does not match MUSIC pair count.")
    if pair_fit_score.size not in (0, pair_rx.size):
        raise ValueError("pair_fit_score length does not match MUSIC pair count.")
    if pair_selection_score.size not in (0, pair_rx.size):
        raise ValueError("pair_selection_score length does not match MUSIC pair count.")

    lookup: Dict[int, Dict[str, np.ndarray]] = {}
    for tx_idx in range(num_tx_total):
        tx_mask = pair_t == int(tx_idx)
        if not np.any(tx_mask):
            lookup[int(tx_idx)] = {
                "rx_detected": np.empty((0,), dtype=int),
                "pair_rx": np.empty((0,), dtype=int),
                "pair_rx_ant": np.empty((0,), dtype=int),
                "selection_score": np.empty((0,), dtype=np.float64),
                "u": np.empty((0, num_tx_ant), dtype=np.complex128),
                "g": np.empty((0,), dtype=np.float64),
            }
            continue

        pair_rx_t = np.asarray(pair_rx[tx_mask], dtype=int)
        pair_rx_ant_t = np.asarray(pair_rx_ant_idx[tx_mask], dtype=int)
        rx_valid = pair_rx_t[(pair_rx_t >= 0) & (pair_rx_t < num_ntn_rx)]
        rx_detected = np.unique(rx_valid)

        selection_score_t = (
            np.asarray(pair_selection_score[tx_mask], dtype=float)
            if pair_selection_score.size > 0
            else np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)
        )
        if not np.any(np.isfinite(selection_score_t)):
            score_user_t = (
                np.asarray(pair_score_user[tx_mask], dtype=float)
                if pair_score_user.size > 0
                else np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)
            )
            fit_score_t = (
                np.asarray(pair_fit_score[tx_mask], dtype=float)
                if pair_fit_score.size > 0
                else np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)
            )
            selection_score_t = np.maximum(
                np.nan_to_num(score_user_t, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            ) * np.maximum(
                np.nan_to_num(fit_score_t, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            )

        u_t = np.asarray(pair_u[tx_mask], dtype=np.complex128)
        alpha_t = np.asarray(pair_alpha_hat[tx_mask], dtype=np.complex128)
        g_t = np.abs(alpha_t) ** 2

        finite_u = np.all(np.isfinite(np.real(u_t)) & np.isfinite(np.imag(u_t)), axis=1)
        finite_g = np.isfinite(g_t)
        keep = finite_u & finite_g

        lookup[int(tx_idx)] = {
            "rx_detected": rx_detected.astype(int),
            "pair_rx": np.asarray(pair_rx_t[keep], dtype=int),
            "pair_rx_ant": np.asarray(pair_rx_ant_t[keep], dtype=int),
            "selection_score": np.asarray(selection_score_t[keep], dtype=float),
            "u": np.asarray(u_t[keep], dtype=np.complex128),
            "g": np.asarray(g_t[keep], dtype=np.float64),
        }

    return lookup


def _extract_tx_music_terms(
    lookup: Dict[str, np.ndarray],
    *,
    num_tx_ant: int,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, np.ndarray]:
    """Collect one TX's deduped MUSIC pair entries for legacy-style est nulling."""
    pair_rx_t = np.asarray(lookup.get("pair_rx", np.empty((0,), dtype=int)), dtype=int)
    pair_rx_ant_t = np.asarray(lookup.get("pair_rx_ant", np.empty((0,), dtype=int)), dtype=int)
    selection_score_t = np.asarray(
        lookup.get("selection_score", np.empty((0,), dtype=float)),
        dtype=float,
    )
    u_t = np.asarray(
        lookup.get("u", np.empty((0, num_tx_ant), dtype=np.complex128)),
        dtype=np.complex128,
    )
    g_t = np.asarray(
        lookup.get("g", np.empty((0,), dtype=np.float64)),
        dtype=np.float64,
    )

    if u_t.ndim == 1:
        if u_t.size == 0:
            u_t = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif u_t.size == num_tx_ant:
            u_t = u_t.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected u shape for one TX: {u_t.shape}")
    if u_t.ndim != 2 or u_t.shape[1] != num_tx_ant:
        raise ValueError(f"u must have shape (K, {num_tx_ant}) for one TX; got {u_t.shape}.")
    if not (pair_rx_t.size == pair_rx_ant_t.size == u_t.shape[0] == g_t.size):
        raise ValueError(
            "Inconsistent per-TX MUSIC lookup lengths: "
            f"pair_rx={pair_rx_t.size}, pair_rx_ant={pair_rx_ant_t.size}, u={u_t.shape[0]}, g={g_t.size}."
        )
    if selection_score_t.size not in (0, g_t.size):
        raise ValueError(
            "Inconsistent per-TX selection score length: "
            f"selection_score={selection_score_t.size}, g={g_t.size}."
        )
    if selection_score_t.size == 0 and g_t.size > 0:
        selection_score_t = np.full((g_t.size,), np.nan, dtype=float)

    pair_rx_t, pair_rx_ant_t, selection_score_t, u_t, g_t = _dedupe_tx_pair_candidates(
        pair_rx_t,
        pair_rx_ant_t,
        selection_score_t,
        u_t,
        g_t,
    )

    max_terms = _resolve_b_term_limit(max_detected_b_terms)
    if max_terms is not None and g_t.size > max_terms:
        rank_score = _pair_rank_score(selection_score_t, g_t)
        keep_idx = np.argsort(-rank_score, kind="mergesort")[:max_terms]
        keep_idx = np.sort(keep_idx)
        pair_rx_t = pair_rx_t[keep_idx]
        pair_rx_ant_t = pair_rx_ant_t[keep_idx]
        selection_score_t = selection_score_t[keep_idx]
        u_t = u_t[keep_idx]
        g_t = g_t[keep_idx]

    return {
        "pair_rx": np.asarray(pair_rx_t, dtype=int),
        "pair_rx_ant": np.asarray(pair_rx_ant_t, dtype=int),
        "selection_score": np.asarray(selection_score_t, dtype=float),
        "u": np.asarray(u_t, dtype=np.complex128),
        "g": np.asarray(g_t, dtype=np.float64),
    }


def _extract_tx_detected_pairs(
    h_ntn_tx: np.ndarray,
    lookup: Dict[str, np.ndarray],
    *,
    num_tx_ant: int,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, np.ndarray]:
    """Collect one TX's detected NTN pair set for true/music-real nulling."""
    h_ntn_t = np.asarray(h_ntn_tx, dtype=np.complex128)
    if h_ntn_t.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")
    if h_ntn_t.shape[2] != int(num_tx_ant):
        raise ValueError(
            f"h_ntn_tx antenna dimension mismatch: expected {num_tx_ant}, got {h_ntn_t.shape[2]}."
        )

    est_inputs = _extract_tx_music_terms(
        lookup,
        num_tx_ant=num_tx_ant,
        max_detected_b_terms=max_detected_b_terms,
    )
    pair_rx_t = np.asarray(est_inputs["pair_rx"], dtype=int)
    pair_rx_ant_t = np.asarray(est_inputs["pair_rx_ant"], dtype=int)
    u_t = np.asarray(est_inputs["u"], dtype=np.complex128)
    g_t = np.asarray(est_inputs["g"], dtype=np.float64)

    valid = (
        (pair_rx_t >= 0)
        & (pair_rx_t < h_ntn_t.shape[0])
        & (pair_rx_ant_t >= 0)
        & (pair_rx_ant_t < h_ntn_t.shape[1])
    )
    if np.any(valid):
        pair_rx_keep = np.asarray(pair_rx_t[valid], dtype=int)
        pair_rx_ant_keep = np.asarray(pair_rx_ant_t[valid], dtype=int)
        u_keep = np.asarray(u_t[valid], dtype=np.complex128)
        g_keep = np.asarray(g_t[valid], dtype=np.float64)
        h_true_keep = np.asarray(
            h_ntn_t[pair_rx_keep, pair_rx_ant_keep, :],
            dtype=np.complex128,
        )
    else:
        pair_rx_keep = np.empty((0,), dtype=int)
        pair_rx_ant_keep = np.empty((0,), dtype=int)
        u_keep = np.empty((0, num_tx_ant), dtype=np.complex128)
        g_keep = np.empty((0,), dtype=np.float64)
        h_true_keep = np.empty((0, num_tx_ant), dtype=np.complex128)

    h_eval = _mask_channel_tensor_to_pairs(h_ntn_t, pair_rx_keep, pair_rx_ant_keep)
    selected_rx = (
        np.unique(pair_rx_keep).astype(int)
        if pair_rx_keep.size > 0
        else np.empty((0,), dtype=int)
    )

    return {
        "pair_rx": pair_rx_keep,
        "pair_rx_ant": pair_rx_ant_keep,
        "selected_rx": selected_rx,
        "u": u_keep,
        "g": g_keep,
        "h_true": h_true_keep,
        "h_eval": h_eval,
    }


def run_small_round(
    h_tn_all: np.ndarray,
    h_ntn_all: np.ndarray,
    *,
    pairs_by_tx: Dict[int, List[Dict[str, Any]]],
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    sim_idx: int | None = None,
    round_idx: int,
    lambda_ranges: Iterable[float] | None = None,
    lambda_ranges_music_est: Iterable[float] | None = None,
    lambda_ranges_music_real: Iterable[float] | None = None,
    tx_power: float,
    snr_noise_power: float,
    inr_noise_power: float,
    max_detected_b_terms: str | int | None = "all",
    print_music_u_corr: bool = False,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Run one small round with three nulling modes on detected NTN pairs."""
    h_tn_all_arr = np.asarray(h_tn_all, dtype=np.complex128)
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    if h_tn_all_arr.ndim != 4:
        raise ValueError("h_tn_all must have shape (num_tn_rx, num_tn_rx_ant, num_tx, num_tx_ant).")
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx, num_tx_ant).")

    num_tn_rx = int(h_tn_all_arr.shape[0])
    num_ntn_rx = int(h_ntn.shape[0])
    num_tx_total = int(h_ntn.shape[2])
    lambda_list = [] if lambda_ranges is None else [float(v) for v in lambda_ranges]
    lambda_list_music_est = (
        [] if lambda_ranges_music_est is None else [float(v) for v in lambda_ranges_music_est]
    )
    lambda_list_music_real = [float(v) for v in (lambda_ranges_music_real or [])]

    raw_snr_db: List[float] = []
    raw_sinr_db: List[float] = []
    true_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    true_sinr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    est_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
    est_sinr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
    music_real_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_real}
    music_real_sinr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_real}
    raw_inr_power = np.zeros((num_ntn_rx,), dtype=np.float64)
    true_inr_power = {lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list}
    est_inr_power = {lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list_music_est}
    music_real_inr_power = {
        lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list_music_real
    }
    detected_rx_mask = np.zeros((num_ntn_rx,), dtype=bool)
    eval_mask = np.zeros((num_ntn_rx,), dtype=bool)
    round_pairs: Dict[int, Dict[str, Any]] = {}
    raw_beams: Dict[int, np.ndarray] = {}
    true_beams: Dict[float, Dict[int, np.ndarray]] = {lambda_: {} for lambda_ in lambda_list}
    est_beams: Dict[float, Dict[int, np.ndarray]] = {lambda_: {} for lambda_ in lambda_list_music_est}
    music_real_beams: Dict[float, Dict[int, np.ndarray]] = {
        lambda_: {} for lambda_ in lambda_list_music_real
    }

    for tx_idx in range(num_tx_total):
        tx_pairs = pairs_by_tx.get(int(tx_idx), [])
        if round_idx >= len(tx_pairs):
            raise ValueError(
                f"round_idx={round_idx} exceeds paired TN count for tx_idx={tx_idx}. "
                "Check min_count scheduling."
            )

        pair = tx_pairs[round_idx]
        h_tn = np.asarray(pair["h_tn"], dtype=np.complex128)
        w_t = np.asarray(pair["w_t"], dtype=np.complex128)
        w_r = np.asarray(pair["w_r"], dtype=np.complex128)
        tn_idx = int(pair["tn_idx"])
        if tn_idx < 0 or tn_idx >= num_tn_rx:
            raise ValueError(f"Invalid TN index for round pair: tn_idx={tn_idx}, num_tn_rx={num_tn_rx}.")
        round_pairs[int(tx_idx)] = {
            "tn_idx": tn_idx,
            "h_tn": h_tn,
            "w_r": w_r,
        }
        raw_beams[int(tx_idx)] = w_t
        raw_snr_db.append(float(pair["snr_raw_db"]))

        h_ntn_tx = np.asarray(h_ntn[:, :, tx_idx, :], dtype=np.complex128)
        lookup = music_lookup.get(int(tx_idx), {})
        rx_detected_t = np.asarray(lookup.get("rx_detected", np.empty((0,), dtype=int)), dtype=int)
        if rx_detected_t.size > 0:
            detected_rx_mask[rx_detected_t] = True

        est_inputs = _extract_tx_music_terms(
            lookup,
            num_tx_ant=h_tn.shape[0],
            max_detected_b_terms=max_detected_b_terms,
        )
        u_t = np.asarray(est_inputs["u"], dtype=np.complex128)
        g_t = np.asarray(est_inputs["g"], dtype=np.float64)

        tx_inputs = _extract_tx_detected_pairs(
            h_ntn_tx,
            lookup,
            num_tx_ant=h_tn.shape[0],
            max_detected_b_terms=max_detected_b_terms,
        )
        selected_rx_t = np.asarray(tx_inputs["selected_rx"], dtype=int)
        if selected_rx_t.size > 0:
            eval_mask[selected_rx_t] = True

        h_ntn_eval_tx = np.asarray(tx_inputs["h_eval"], dtype=np.complex128)
        h_true_t = np.asarray(tx_inputs["h_true"], dtype=np.complex128)

        interference_term_true = _covariance_from_channel_vectors(
            h_true_t,
            num_tx_ant=h_tn.shape[0],
        )
        u_true_t, g_true_t = _channel_vectors_to_noncoh_terms(
            h_true_t,
            num_tx_ant=h_tn.shape[0],
            eps=eps,
        )
        raw_inr_power += _interference_power_per_rx(h_ntn_tx, w_t)

        for lambda_ in lambda_list:
            v_null_true, _, _, _ = nulling_bf(h_tn, w_r, interference_term_true, lambda_)
            true_beams[lambda_][int(tx_idx)] = np.asarray(v_null_true, dtype=np.complex128)

            true_snr_linear = (
                np.abs((v_null_true.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )

            true_snr_db[lambda_].append(float(_safe_db(true_snr_linear, eps=eps)))
            true_inr_power[lambda_] += _interference_power_per_rx(h_ntn_eval_tx, v_null_true)

        for lambda_ in lambda_list_music_est:
            v_null_est, _, _, _ = nulling_bf_music_noncoh(h_tn, w_r, u_t, g_t, lambda_, eps=eps)
            est_beams[lambda_][int(tx_idx)] = np.asarray(v_null_est, dtype=np.complex128)
            est_snr_linear = (
                np.abs((v_null_est.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )
            est_snr_db[lambda_].append(float(_safe_db(est_snr_linear, eps=eps)))
            est_inr_power[lambda_] += _interference_power_per_rx(h_ntn_tx, v_null_est)

        for lambda_ in lambda_list_music_real:
            v_null_music_real, _, _, _ = nulling_bf_music_noncoh(h_tn, w_r, u_true_t, g_true_t,  lambda_, eps=eps)
            music_real_beams[lambda_][int(tx_idx)] = np.asarray(v_null_music_real, dtype=np.complex128)
            music_real_snr_linear = (
                np.abs((v_null_music_real.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )
            music_real_snr_db[lambda_].append(float(_safe_db(music_real_snr_linear, eps=eps)))
            music_real_inr_power[lambda_] += _interference_power_per_rx(h_ntn_eval_tx, v_null_music_real)

    for tx_idx in range(num_tx_total):
        pair = round_pairs[int(tx_idx)]
        tn_idx = int(pair["tn_idx"])
        h_tn = np.asarray(pair["h_tn"], dtype=np.complex128)
        w_r = np.asarray(pair["w_r"], dtype=np.complex128)

        desired_raw = _tn_link_power(h_tn, raw_beams[int(tx_idx)], w_r)
        interf_raw = 0.0
        for other_tx in range(num_tx_total):
            if int(other_tx) == int(tx_idx):
                continue
            h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
            interf_raw += _tn_link_power(h_interf, raw_beams[int(other_tx)], w_r)
        raw_sinr_linear = desired_raw * float(tx_power) / (
            float(snr_noise_power) + interf_raw * float(tx_power)
        )
        raw_sinr_db.append(float(_safe_db(raw_sinr_linear, eps=eps)))

        for lambda_ in lambda_list:
            beam_true = true_beams[lambda_][int(tx_idx)]
            desired_true = _tn_link_power(h_tn, beam_true, w_r)
            interf_true = 0.0
            for other_tx in range(num_tx_total):
                if int(other_tx) == int(tx_idx):
                    continue
                h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
                interf_true += _tn_link_power(h_interf, true_beams[lambda_][int(other_tx)], w_r)
            true_sinr_linear = desired_true * float(tx_power) / (
                float(snr_noise_power) + interf_true * float(tx_power)
            )
            true_sinr_db[lambda_].append(float(_safe_db(true_sinr_linear, eps=eps)))

        for lambda_ in lambda_list_music_est:
            beam_est = est_beams[lambda_][int(tx_idx)]
            desired_est = _tn_link_power(h_tn, beam_est, w_r)
            interf_est = 0.0
            for other_tx in range(num_tx_total):
                if int(other_tx) == int(tx_idx):
                    continue
                h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
                interf_est += _tn_link_power(h_interf, est_beams[lambda_][int(other_tx)], w_r)
            est_sinr_linear = desired_est * float(tx_power) / (
                float(snr_noise_power) + interf_est * float(tx_power)
            )
            est_sinr_db[lambda_].append(float(_safe_db(est_sinr_linear, eps=eps)))

        for lambda_ in lambda_list_music_real:
            beam_music_real = music_real_beams[lambda_][int(tx_idx)]
            desired_music_real = _tn_link_power(h_tn, beam_music_real, w_r)
            interf_music_real = 0.0
            for other_tx in range(num_tx_total):
                if int(other_tx) == int(tx_idx):
                    continue
                h_interf = np.asarray(h_tn_all_arr[tn_idx, :, other_tx, :], dtype=np.complex128).T
                interf_music_real += _tn_link_power(
                    h_interf,
                    music_real_beams[lambda_][int(other_tx)],
                    w_r,
                )
            music_real_sinr_linear = desired_music_real * float(tx_power) / (
                float(snr_noise_power) + interf_music_real * float(tx_power)
            )
            music_real_sinr_db[lambda_].append(float(_safe_db(music_real_sinr_linear, eps=eps)))

    raw_inr_db = (
        _safe_db(raw_inr_power[detected_rx_mask] * float(tx_power) / float(inr_noise_power), eps=eps)
        if np.any(detected_rx_mask)
        else np.empty((0,), dtype=np.float64)
    )
    true_inr_db = {
        lambda_: (
            _safe_db(
                true_inr_power[lambda_][eval_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(eval_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list
    }
    est_inr_db = {
        lambda_: (
            _safe_db(
                est_inr_power[lambda_][detected_rx_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(detected_rx_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list_music_est
    }
    music_real_inr_db = {
        lambda_: (
            _safe_db(
                music_real_inr_power[lambda_][eval_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(eval_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list_music_real
    }

    return {
        "raw_snr_db": np.asarray(raw_snr_db, dtype=np.float64),
        "raw_sinr_db": np.asarray(raw_sinr_db, dtype=np.float64),
        "true_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_snr_db.items()},
        "true_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_sinr_db.items()
        },
        "est_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_db.items()},
        "est_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_sinr_db.items()
        },
        "music_real_snr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_snr_db.items()
        },
        "music_real_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_sinr_db.items()
        },
        "raw_inr_db": np.asarray(raw_inr_db, dtype=np.float64),
        "true_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_inr_db.items()},
        "est_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_db.items()},
        "music_real_inr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_inr_db.items()
        },
        "detected_mask": detected_rx_mask,
        "detected_count": int(np.count_nonzero(detected_rx_mask)),
    }


def run_nulling_cdf_experiment(
    scene_config: Any,
    *,
    num_macro_sims: int,
    compute_positions_kwargs: Dict[str, Any],
    compute_paths_kwargs: Dict[str, Any],
    lambda_ranges: Iterable[float] | None = None,
    lambda_ranges_music_est: Iterable[float] | None = None,
    lambda_ranges_music_real: Iterable[float] | None = None,
    h_tn_th: float,
    tx_antennas: int,
    tx_power: float,
    snr_noise_power: float,
    inr_noise_power: float,
    music_kwargs: Dict[str, Any],
    sionna_phi_is_global: bool = True,
    theta_display_mode: str = "elevation",
    eps: float = 1e-12,
    plot_first_sim_only: bool = True,
    show_progress: bool = True,
    print_music_u_corr: bool = True,
    resample_satellite_per_macro: bool = False,
    satellite_azimuth_range_deg: Tuple[float, float] = (0.0, 360.0),
    satellite_elevation_range_deg: Tuple[float, float] = (35.0, 90.0),
    satellite_rng_seed: int | None = None,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, Any]:
    """Run the nulling CDF experiment across macro simulations."""
    if int(num_macro_sims) <= 0:
        raise ValueError("num_macro_sims must be positive.")
    resolved_b_term_limit = _resolve_b_term_limit(max_detected_b_terms)

    try:
        from tqdm.auto import trange
    except Exception:
        trange = None

    lambda_list = [] if lambda_ranges is None else [float(v) for v in lambda_ranges]
    lambda_list_music_est = (
        [float(v) for v in lambda_ranges_music_est]
        if lambda_ranges_music_est is not None
        else list(lambda_list)
    )
    lambda_list_music_real = [float(v) for v in (lambda_ranges_music_real or [])]
    raw_snr_all: List[float] = []
    raw_sinr_all: List[float] = []
    raw_inr_all: List[float] = []
    true_snr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    true_sinr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    est_snr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
    est_sinr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
    music_real_snr_all: Dict[float, List[float]] = {
        lambda_: [] for lambda_ in lambda_list_music_real
    }
    music_real_sinr_all: Dict[float, List[float]] = {
        lambda_: [] for lambda_ in lambda_list_music_real
    }
    true_inr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    est_inr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list_music_est}
    music_real_inr_all: Dict[float, List[float]] = {
        lambda_: [] for lambda_ in lambda_list_music_real
    }
    macro_stats: List[Dict[str, Any]] = []
    bs_pos_ref: np.ndarray | None = None
    sat_rng = (
        np.random.default_rng(satellite_rng_seed)
        if bool(resample_satellite_per_macro)
        else None
    )

    iterator = (
        trange(int(num_macro_sims), desc="Monte Carlo", leave=False)
        if show_progress and trange is not None
        else range(int(num_macro_sims))
    )

    for sim_idx in iterator:
        pos_kwargs = dict(compute_positions_kwargs)
        if plot_first_sim_only and sim_idx > 0:
            for key in ("plot_grid", "plot_bs", "plot_tn", "plot_ntn"):
                if key in pos_kwargs:
                    pos_kwargs[key] = False

        if sat_rng is not None:
            sat_azimuth_deg = float(
                sat_rng.uniform(
                    float(satellite_azimuth_range_deg[0]),
                    float(satellite_azimuth_range_deg[1]),
                )
            )
            sat_elevation_deg = float(
                sat_rng.uniform(
                    float(satellite_elevation_range_deg[0]),
                    float(satellite_elevation_range_deg[1]),
                )
            )
            pos_kwargs["azimuth"] = sat_azimuth_deg
            pos_kwargs["elevation"] = sat_elevation_deg
        else:
            sat_azimuth_deg = float(pos_kwargs["azimuth"])
            sat_elevation_deg = float(pos_kwargs["elevation"])

        scene_config.compute_positions(**pos_kwargs)
        tx_pos = np.asarray(scene_config.tx_pos, dtype=np.float64)
        sat_look_pos = np.asarray(scene_config.ntn_look_pos, dtype=np.float64).copy()
        if bs_pos_ref is None:
            bs_pos_ref = tx_pos.copy()
        elif tx_pos.shape != bs_pos_ref.shape or not np.allclose(tx_pos, bs_pos_ref):
            raise RuntimeError(
                "BS positions changed across macro simulations. "
                "The requested experiment assumes fixed BS positions."
            )

        scene_config.compute_paths(**compute_paths_kwargs)
        h_tn_all = collapse_cir_to_narrowband(scene_config.a_tn)
        h_ntn_all = collapse_cir_to_narrowband(scene_config.a_ntn)

        pairing = pair_tn_to_strongest_tx(
            h_tn_all,
            h_tn_th=float(h_tn_th),
            tx_antennas=int(tx_antennas),
            tx_power=float(tx_power),
            snr_noise_power=float(snr_noise_power),
            eps=eps,
        )
        min_count = int(pairing["min_count"])
        num_ntn_rx = int(h_ntn_all.shape[0])
        num_tx_total = int(h_ntn_all.shape[2])
        num_tx_ant = int(h_ntn_all.shape[3])

        ntn_music_out = run_music_standard_pipeline(
            h_ntn_all,
            **music_kwargs,
        )
        ntn_truth = build_ntn_truth_from_paths(
            scene_config.paths_ntn,
            scene_config.a_ntn,
            num_tx_total=num_tx_total,
            nsect=int(music_kwargs["nsect"]),
            sionna_phi_is_global=bool(sionna_phi_is_global),
        )
        music_quality = summarize_ntn_music_quality(
            h_ntn_all,
            ntn_music_out,
            ntn_truth["pair_map"],
            theta_display_mode=str(theta_display_mode),
            eps=eps,
        )
        music_lookup = build_music_tx_lookup(
            ntn_music_out,
            num_ntn_rx=num_ntn_rx,
            num_tx_total=num_tx_total,
            num_tx_ant=num_tx_ant,
        )

        detected_rx_union = np.asarray(ntn_music_out.get("detected_rx_indices_unique", []), dtype=int)
        interfered_ntn_count = int(np.count_nonzero(np.any(np.abs(h_ntn_all) > eps, axis=(1, 2, 3))))
        pair_counts_by_tx = np.asarray(pairing["pair_counts_by_tx"], dtype=int)
        noncoh_metrics = summarize_music_noncoh_quality(
            h_ntn_all,
            music_lookup,
            max_detected_b_terms=resolved_b_term_limit,
            eps=eps,
        )
        angle_metrics = music_quality["angle_metrics"]
        if print_music_u_corr:
            print(
                "[MUSIC sim] "
                f"sim={int(sim_idx)} "
                f"sat_az_deg={float(sat_azimuth_deg):.2f} sat_el_deg={float(sat_elevation_deg):.2f} "
                f"det_ntn={int(detected_rx_union.size)} interf_ntn={int(interfered_ntn_count)} "
                f"matched_pairs={int(angle_metrics.get('matched_pairs', 0))} "
                f"u_pairs={int(noncoh_metrics['pairs'])} tx_used={int(noncoh_metrics['tx_with_pairs'])} "
                f"phi_mae_deg={float(angle_metrics.get('phi_mae_deg', np.nan)):.3f} "
                f"elev_mae_deg={float(angle_metrics.get('elev_mae_deg', np.nan)):.3f} "
                f"u_rho_mean={float(noncoh_metrics['u_rho_mean']):.6f} "
                f"u_err_mean={float(noncoh_metrics['u_err_mean']):.6e} "
                f"g_rel_err_mean={float(noncoh_metrics['g_rel_err_mean']):.6e}"
            )

        macro_stats.append(
            {
                "sim_idx": int(sim_idx),
                "min_count": int(min_count),
                "pair_counts_by_tx": pair_counts_by_tx.copy(),
                "detected_ntn_count": int(detected_rx_union.size),
                "interfered_ntn_count": interfered_ntn_count,
                "satellite_azimuth_deg": float(sat_azimuth_deg),
                "satellite_elevation_deg": float(sat_elevation_deg),
                "satellite_look_pos": sat_look_pos.copy(),
                "angle_metrics": angle_metrics,
                "detected_subset_metrics": music_quality["detected_subset_metrics"],
                "detected_pairs_summary": music_quality["detected_pairs_summary"],
                "noncoh_metrics": noncoh_metrics,
            }
        )

        if min_count <= 0:
            continue

        for round_idx in range(min_count):
            round_out = run_small_round(
                h_tn_all,
                h_ntn_all,
                pairs_by_tx=pairing["pairs_by_tx"],
                music_lookup=music_lookup,
                sim_idx=int(sim_idx),
                round_idx=int(round_idx),
                lambda_ranges=lambda_list,
                lambda_ranges_music_est=lambda_list_music_est,
                lambda_ranges_music_real=lambda_list_music_real,
                tx_power=float(tx_power),
                snr_noise_power=float(snr_noise_power),
                inr_noise_power=float(inr_noise_power),
                max_detected_b_terms=resolved_b_term_limit,
                print_music_u_corr=bool(print_music_u_corr),
                eps=eps,
            )

            raw_snr_all.extend(np.asarray(round_out["raw_snr_db"], dtype=np.float64).tolist())
            raw_sinr_all.extend(np.asarray(round_out["raw_sinr_db"], dtype=np.float64).tolist())
            raw_inr_all.extend(np.asarray(round_out["raw_inr_db"], dtype=np.float64).tolist())
            for lambda_ in lambda_list:
                true_snr_all[lambda_].extend(
                    np.asarray(round_out["true_snr_db"][lambda_], dtype=np.float64).tolist()
                )
                true_sinr_all[lambda_].extend(
                    np.asarray(round_out["true_sinr_db"][lambda_], dtype=np.float64).tolist()
                )
                true_inr_all[lambda_].extend(
                    np.asarray(round_out["true_inr_db"][lambda_], dtype=np.float64).tolist()
                )
            for lambda_ in lambda_list_music_est:
                est_snr_all[lambda_].extend(
                    np.asarray(round_out["est_snr_db"][lambda_], dtype=np.float64).tolist()
                )
                est_sinr_all[lambda_].extend(
                    np.asarray(round_out["est_sinr_db"][lambda_], dtype=np.float64).tolist()
                )
                est_inr_all[lambda_].extend(
                    np.asarray(round_out["est_inr_db"][lambda_], dtype=np.float64).tolist()
                )
            for lambda_ in lambda_list_music_real:
                music_real_snr_all[lambda_].extend(
                    np.asarray(round_out["music_real_snr_db"][lambda_], dtype=np.float64).tolist()
                )
                music_real_sinr_all[lambda_].extend(
                    np.asarray(round_out["music_real_sinr_db"][lambda_], dtype=np.float64).tolist()
                )
                music_real_inr_all[lambda_].extend(
                    np.asarray(round_out["music_real_inr_db"][lambda_], dtype=np.float64).tolist()
                )

    return {
        "raw_snr_db": np.asarray(raw_snr_all, dtype=np.float64),
        "raw_sinr_db": np.asarray(raw_sinr_all, dtype=np.float64),
        "raw_inr_db": np.asarray(raw_inr_all, dtype=np.float64),
        "true_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_snr_all.items()},
        "true_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_sinr_all.items()
        },
        "est_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_all.items()},
        "est_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_sinr_all.items()
        },
        "music_real_snr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_snr_all.items()
        },
        "music_real_sinr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_sinr_all.items()
        },
        "true_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in true_inr_all.items()},
        "est_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_all.items()},
        "music_real_inr_db": {
            lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in music_real_inr_all.items()
        },
        "macro_stats": macro_stats,
        "bs_pos_ref": bs_pos_ref,
        "lambda_ranges": np.asarray(lambda_list, dtype=np.float64),
        "lambda_ranges_music_est": np.asarray(lambda_list_music_est, dtype=np.float64),
        "lambda_ranges_music_real": np.asarray(lambda_list_music_real, dtype=np.float64),
        "max_detected_b_terms": "all" if resolved_b_term_limit is None else int(resolved_b_term_limit),
    }


def save_experiment_metrics(
    experiment_out: Dict[str, Any],
    *,
    result_dir: str | Path = "result",
    output_name: str = "nulling_cdf_metrics.npz",
) -> Path:
    """Save experiment arrays and macro statistics for later reuse."""
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    save_path = result_path / output_name

    save_dict: Dict[str, Any] = {
        "raw_snr_db": np.asarray(experiment_out["raw_snr_db"], dtype=np.float64),
        "raw_sinr_db": np.asarray(experiment_out.get("raw_sinr_db", np.empty((0,), dtype=np.float64)), dtype=np.float64),
        "raw_inr_db": np.asarray(experiment_out["raw_inr_db"], dtype=np.float64),
        "lambda_ranges": np.asarray(experiment_out["lambda_ranges"], dtype=np.float64),
        "lambda_ranges_music_est": np.asarray(
            experiment_out.get("lambda_ranges_music_est", np.empty((0,), dtype=np.float64)),
            dtype=np.float64,
        ),
        "lambda_ranges_music_real": np.asarray(
            experiment_out.get("lambda_ranges_music_real", np.empty((0,), dtype=np.float64)),
            dtype=np.float64,
        ),
        "max_detected_b_terms": np.asarray([str(experiment_out.get("max_detected_b_terms", "all"))]),
    }
    if experiment_out.get("bs_pos_ref") is not None:
        save_dict["bs_pos_ref"] = np.asarray(experiment_out["bs_pos_ref"], dtype=np.float64)
    for lambda_, vals in experiment_out["true_snr_db"].items():
        save_dict[f"true_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("true_sinr_db", {}).items():
        save_dict[f"true_sinr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["est_snr_db"].items():
        save_dict[f"est_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("est_sinr_db", {}).items():
        save_dict[f"est_sinr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["true_inr_db"].items():
        save_dict[f"true_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["est_inr_db"].items():
        save_dict[f"est_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("music_real_snr_db", {}).items():
        save_dict[f"music_real_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("music_real_sinr_db", {}).items():
        save_dict[f"music_real_sinr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out.get("music_real_inr_db", {}).items():
        save_dict[f"music_real_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)

    macro_stats = experiment_out.get("macro_stats", [])
    save_dict["macro_stats_sim_idx"] = np.asarray([row["sim_idx"] for row in macro_stats], dtype=int)
    save_dict["macro_stats_min_count"] = np.asarray([row["min_count"] for row in macro_stats], dtype=int)
    save_dict["macro_stats_detected_ntn_count"] = np.asarray(
        [row["detected_ntn_count"] for row in macro_stats],
        dtype=int,
    )
    save_dict["macro_stats_interfered_ntn_count"] = np.asarray(
        [row["interfered_ntn_count"] for row in macro_stats],
        dtype=int,
    )
    if macro_stats:
        save_dict["macro_stats_satellite_azimuth_deg"] = np.asarray(
            [row.get("satellite_azimuth_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_satellite_elevation_deg"] = np.asarray(
            [row.get("satellite_elevation_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_satellite_look_pos"] = np.stack(
            [
                np.asarray(row.get("satellite_look_pos", [np.nan, np.nan, np.nan]), dtype=np.float64)
                for row in macro_stats
            ],
            axis=0,
        )
        save_dict["macro_stats_pair_counts_by_tx"] = np.stack(
            [np.asarray(row["pair_counts_by_tx"], dtype=int) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_phi_mae_deg"] = np.asarray(
            [row.get("angle_metrics", {}).get("phi_mae_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_elev_mae_deg"] = np.asarray(
            [row.get("angle_metrics", {}).get("elev_mae_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_angle_match_count"] = np.asarray(
            [row.get("angle_metrics", {}).get("matched_pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_detected_subset_count"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("count", 0) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_nrmse"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("nrmse", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_cos_sim"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("cos_sim", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_mag_mae"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("mag_mae", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_power_ratio_db"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("power_ratio_db", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_count"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_detected_pairs_nrmse_mean"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("nrmse_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_nrmse_median"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("nrmse_median", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_cos_mean"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("cos_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_cos_median"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("cos_median", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_noncoh_pairs"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_noncoh_tx_with_pairs"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("tx_with_pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_noncoh_u_rho_mean"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("u_rho_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_noncoh_u_err_mean"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("u_err_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_noncoh_g_rel_err_mean"] = np.asarray(
            [row.get("noncoh_metrics", {}).get("g_rel_err_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )

    np.savez(save_path, **save_dict)
    return save_path
