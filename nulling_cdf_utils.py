from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from BeamformingCalc import nulling_bf, svd_bf
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
    """Per-NTN received interference power for one TX beam.

    Parameters
    ----------
    h_ntn_tx : np.ndarray
        Shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).
    beam : np.ndarray
        Shape (num_tx_ant, 1) or (num_tx_ant,).
    """
    h = np.asarray(h_ntn_tx, dtype=np.complex128)
    v = np.asarray(beam, dtype=np.complex128).reshape(-1, 1)
    if h.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")
    if h.shape[2] != v.shape[0]:
        raise ValueError(
            f"Beam dimension mismatch: h_ntn_tx has {h.shape[2]} TX antennas, beam has {v.shape[0]}."
        )
    eff = np.einsum("nra,ab->nr", h, v, optimize=True)
    return np.sum(np.abs(eff) ** 2, axis=1).real.astype(np.float64)


def _covariance_from_channel_vectors(
    h_hat_vectors: np.ndarray,
    *,
    num_tx_ant: int,
) -> np.ndarray:
    """Build sum_k h_hat_k h_hat_k^H from reconstructed channel vectors."""
    h_hat = np.asarray(h_hat_vectors, dtype=np.complex128)
    if h_hat.size == 0:
        return np.zeros((num_tx_ant, num_tx_ant), dtype=np.complex128)
    if h_hat.ndim == 1:
        h_hat = h_hat.reshape(1, -1)
    if h_hat.ndim != 2 or h_hat.shape[1] != num_tx_ant:
        raise ValueError(
            f"h_hat_vectors must have shape (K, {num_tx_ant}); got {h_hat.shape}."
        )
    return np.einsum("ka,kb->ab", h_hat, np.conjugate(h_hat), optimize=True)


def _resolve_b_term_limit(max_detected_b_terms: str | int | None) -> int | None:
    """Normalize the user-facing B-term limit.

    Returns
    -------
    int | None
        `None` means keep all detected terms. A positive integer means keep at
        most that many detected terms for each TX.
    """
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


def pair_tn_to_strongest_tx(
    h_tn_all: np.ndarray,
    *,
    h_tn_th: float,
    tx_antennas: int,
    tx_power: float,
    snr_noise_power: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Pair each TN to its strongest valid TX over all BS sectors.

    Pairing rule:
    1. Remove zero channels.
    2. Keep only TXs with Frobenius norm > h_tn_th.
    3. Each TN is paired to the strongest remaining TX only.
    """
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
    """Collect detected NTN reconstructions per TX.

    The returned `h_hat` uses the raw-channel convention and is reconstructed as
    `h_hat = alpha_hat_raw * u_hat_raw`, matching the old covariance-based nulling
    path that expects channel vectors rather than only steering directions.
    """
    pair_rx = np.asarray(ntn_music_out.get("pair_rx_idx", []), dtype=int)
    pair_t = np.asarray(ntn_music_out.get("pair_t_idx", []), dtype=int)
    pair_u = np.asarray(ntn_music_out.get("pair_u_hat", []), dtype=np.complex128)
    pair_u_raw = np.asarray(
        ntn_music_out.get("pair_u_hat_raw", ntn_music_out.get("pair_u_hat", [])),
        dtype=np.complex128,
    )
    pair_alpha_hat = np.asarray(ntn_music_out.get("pair_alpha_hat_raw", []), dtype=np.complex128)
    pair_h_hat_raw = np.asarray(ntn_music_out.get("pair_h_hat_vec_raw", []), dtype=np.complex128)
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

    if pair_u_raw.ndim == 1:
        if pair_u_raw.size == 0:
            pair_u_raw = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif pair_u_raw.size == num_tx_ant:
            pair_u_raw = pair_u_raw.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected pair_u_hat_raw shape: {pair_u_raw.shape}")
    if pair_u_raw.ndim != 2:
        raise ValueError(f"pair_u_hat_raw must be 2D after reshape, got {pair_u_raw.shape}")
    if pair_u_raw.shape[1] != num_tx_ant and pair_u_raw.shape[0] > 0:
        raise ValueError(
            f"pair_u_hat_raw antenna dimension mismatch: expected {num_tx_ant}, got {pair_u_raw.shape[1]}."
        )

    if pair_h_hat_raw.ndim == 1:
        if pair_h_hat_raw.size == 0:
            pair_h_hat_raw = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif pair_h_hat_raw.size == num_tx_ant:
            pair_h_hat_raw = pair_h_hat_raw.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected pair_h_hat_vec_raw shape: {pair_h_hat_raw.shape}")
    if pair_h_hat_raw.ndim != 2:
        raise ValueError(f"pair_h_hat_vec_raw must be 2D after reshape, got {pair_h_hat_raw.shape}")
    if pair_h_hat_raw.shape[1] != num_tx_ant and pair_h_hat_raw.shape[0] > 0:
        raise ValueError(
            "pair_h_hat_vec_raw antenna dimension mismatch: "
            f"expected {num_tx_ant}, got {pair_h_hat_raw.shape[1]}."
        )

    if not (
        pair_rx.size
        == pair_t.size
        == pair_rx_ant_idx.size
        == pair_alpha_hat.size
        == pair_u.shape[0]
        == pair_u_raw.shape[0]
    ):
        raise ValueError("Inconsistent MUSIC pair lengths in ntn_music_out.")
    if pair_h_hat_raw.shape[0] not in (0, pair_rx.size):
        raise ValueError("pair_h_hat_vec_raw length does not match MUSIC pair count.")
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
                "h_hat": np.empty((0, num_tx_ant), dtype=np.complex128),
            }
            continue

        rx_t = pair_rx[tx_mask]
        rx_t = rx_t[(rx_t >= 0) & (rx_t < num_ntn_rx)]
        rx_detected = np.unique(rx_t)

        pair_rx_t = np.asarray(pair_rx[tx_mask], dtype=int)
        pair_rx_ant_t = np.asarray(pair_rx_ant_idx[tx_mask], dtype=int)
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
        selection_score_t = (
            np.asarray(pair_selection_score[tx_mask], dtype=float)
            if pair_selection_score.size > 0
            else np.full((int(np.count_nonzero(tx_mask)),), np.nan, dtype=float)
        )
        u_t = np.asarray(pair_u[tx_mask], dtype=np.complex128)
        u_raw_t = np.asarray(pair_u_raw[tx_mask], dtype=np.complex128)
        alpha_t = np.asarray(pair_alpha_hat[tx_mask], dtype=np.complex128)
        h_hat_raw_t = (
            np.asarray(pair_h_hat_raw[tx_mask], dtype=np.complex128)
            if pair_h_hat_raw.shape[0] > 0
            else np.empty((0, num_tx_ant), dtype=np.complex128)
        )

        finite_u = np.all(np.isfinite(np.real(u_t)) & np.isfinite(np.imag(u_t)), axis=1)
        finite_u_raw = np.all(np.isfinite(np.real(u_raw_t)) & np.isfinite(np.imag(u_raw_t)), axis=1)
        finite_a = np.isfinite(np.real(alpha_t)) & np.isfinite(np.imag(alpha_t))
        keep = finite_u & finite_u_raw & finite_a

        if np.any(keep):
            pair_rx_keep = pair_rx_t[keep]
            pair_rx_ant_keep = pair_rx_ant_t[keep]
            score_user_keep = score_user_t[keep]
            fit_score_keep = fit_score_t[keep]
            selection_score_keep = selection_score_t[keep]
            u_raw_keep = u_raw_t[keep]
            alpha_keep = alpha_t[keep]
            h_hat_keep = alpha_keep[:, None] * u_raw_keep

            if h_hat_raw_t.shape[0] > 0:
                h_hat_raw_keep = h_hat_raw_t[keep]
                finite_h_hat_raw = np.all(
                    np.isfinite(np.real(h_hat_raw_keep)) & np.isfinite(np.imag(h_hat_raw_keep)),
                    axis=1,
                )
                if np.any(finite_h_hat_raw):
                    h_hat_keep[finite_h_hat_raw] = h_hat_raw_keep[finite_h_hat_raw]
            if not np.any(np.isfinite(selection_score_keep)):
                fit_nonneg = np.maximum(
                    np.nan_to_num(fit_score_keep, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                )
                score_nonneg = np.maximum(
                    np.nan_to_num(score_user_keep, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                )
                selection_score_keep = score_nonneg * fit_nonneg
        else:
            pair_rx_keep = np.empty((0,), dtype=int)
            pair_rx_ant_keep = np.empty((0,), dtype=int)
            score_user_keep = np.empty((0,), dtype=float)
            fit_score_keep = np.empty((0,), dtype=float)
            selection_score_keep = np.empty((0,), dtype=float)
            h_hat_keep = np.empty((0, num_tx_ant), dtype=np.complex128)

        lookup[int(tx_idx)] = {
            "rx_detected": rx_detected.astype(int),
            "pair_rx": np.asarray(pair_rx_keep, dtype=int),
            "pair_rx_ant": np.asarray(pair_rx_ant_keep, dtype=int),
            "score_user": np.asarray(score_user_keep, dtype=float),
            "fit_score": np.asarray(fit_score_keep, dtype=float),
            "selection_score": np.asarray(selection_score_keep, dtype=float),
            "h_hat": np.asarray(h_hat_keep, dtype=np.complex128),
        }

    return lookup


def _extract_tx_b_inputs(
    h_ntn_tx: np.ndarray,
    lookup: Dict[str, np.ndarray],
    *,
    num_tx_ant: int,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, np.ndarray]:
    """Collect the per-TX channel vectors that are actually used to form B."""
    h_ntn_t = np.asarray(h_ntn_tx, dtype=np.complex128)
    if h_ntn_t.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")
    if h_ntn_t.shape[2] != int(num_tx_ant):
        raise ValueError(
            f"h_ntn_tx antenna dimension mismatch: expected {num_tx_ant}, got {h_ntn_t.shape[2]}."
        )

    pair_rx_t = np.asarray(lookup.get("pair_rx", np.empty((0,), dtype=int)), dtype=int)
    pair_rx_ant_t = np.asarray(lookup.get("pair_rx_ant", np.empty((0,), dtype=int)), dtype=int)
    selection_score_t = np.asarray(
        lookup.get("selection_score", np.empty((0,), dtype=float)),
        dtype=float,
    )
    h_hat_t = np.asarray(
        lookup.get("h_hat", np.empty((0, num_tx_ant), dtype=np.complex128)),
        dtype=np.complex128,
    )

    if h_hat_t.ndim == 1:
        if h_hat_t.size == 0:
            h_hat_t = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif h_hat_t.size == num_tx_ant:
            h_hat_t = h_hat_t.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected h_hat shape for one TX: {h_hat_t.shape}")
    if h_hat_t.ndim != 2 or h_hat_t.shape[1] != num_tx_ant:
        raise ValueError(
            f"h_hat must have shape (K, {num_tx_ant}) for one TX; got {h_hat_t.shape}."
        )
    if not (pair_rx_t.size == pair_rx_ant_t.size == h_hat_t.shape[0]):
        raise ValueError(
            "Inconsistent per-TX MUSIC lookup lengths: "
            f"pair_rx={pair_rx_t.size}, pair_rx_ant={pair_rx_ant_t.size}, h_hat={h_hat_t.shape[0]}."
        )
    if selection_score_t.size not in (0, h_hat_t.shape[0]):
        raise ValueError(
            "Inconsistent per-TX selection score length: "
            f"selection_score={selection_score_t.size}, h_hat={h_hat_t.shape[0]}."
        )
    if selection_score_t.size == 0 and h_hat_t.shape[0] > 0:
        selection_score_t = np.full((h_hat_t.shape[0],), np.nan, dtype=float)

    max_terms = _resolve_b_term_limit(max_detected_b_terms)
    if max_terms is not None and h_hat_t.shape[0] > max_terms:
        # Keep the most reliable detected users for this TX. The primary ranking
        # uses a combined MUSIC reliability score = detection score * steering-fit
        # score. If that score is unavailable, fall back to reconstructed channel
        # energy. The same selected indices are used for both true-h and h_hat
        # branches so the comparison stays apples-to-apples.
        rank_score = np.asarray(selection_score_t, dtype=float)
        if not np.any(np.isfinite(rank_score)):
            rank_score = np.sum(np.abs(h_hat_t) ** 2, axis=1).real
        else:
            invalid = ~np.isfinite(rank_score)
            if np.any(invalid):
                fallback_power = np.sum(np.abs(h_hat_t[invalid]) ** 2, axis=1).real
                rank_score = rank_score.copy()
                rank_score[invalid] = fallback_power
        keep_idx = np.argsort(-rank_score, kind="mergesort")[:max_terms]
        keep_idx = np.sort(keep_idx)
        pair_rx_t = pair_rx_t[keep_idx]
        pair_rx_ant_t = pair_rx_ant_t[keep_idx]
        selection_score_t = selection_score_t[keep_idx]
        h_hat_t = h_hat_t[keep_idx]

    valid_true_pairs = (
        (pair_rx_t >= 0)
        & (pair_rx_t < h_ntn_t.shape[0])
        & (pair_rx_ant_t >= 0)
        & (pair_rx_ant_t < h_ntn_t.shape[1])
    )
    if np.any(valid_true_pairs):
        h_true_t = np.asarray(
            h_ntn_t[pair_rx_t[valid_true_pairs], pair_rx_ant_t[valid_true_pairs], :],
            dtype=np.complex128,
        )
    else:
        h_true_t = np.empty((0, num_tx_ant), dtype=np.complex128)

    return {
        "pair_rx": pair_rx_t,
        "pair_rx_ant": pair_rx_ant_t,
        "selection_score": selection_score_t,
        "valid_true_pairs": valid_true_pairs,
        "selected_rx": np.unique(pair_rx_t).astype(int) if pair_rx_t.size > 0 else np.empty((0,), dtype=int),
        "h_true": h_true_t,
        "h_hat": h_hat_t,
    }


def summarize_b_usage_by_tx(
    h_ntn_all: np.ndarray,
    *,
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, np.ndarray]:
    """Summarize how many NTN channels each TX actually uses to build B."""
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx, num_tx_ant).")

    num_tx_total = int(h_ntn.shape[2])
    num_tx_ant = int(h_ntn.shape[3])
    b_true_ntn_count_by_tx = np.zeros((num_tx_total,), dtype=int)
    b_true_term_count_by_tx = np.zeros((num_tx_total,), dtype=int)
    b_hat_ntn_count_by_tx = np.zeros((num_tx_total,), dtype=int)
    b_hat_term_count_by_tx = np.zeros((num_tx_total,), dtype=int)

    for tx_idx in range(num_tx_total):
        tx_inputs = _extract_tx_b_inputs(
            h_ntn[:, :, tx_idx, :],
            music_lookup.get(int(tx_idx), {}),
            num_tx_ant=num_tx_ant,
            max_detected_b_terms=max_detected_b_terms,
        )
        true_pair_rx = np.asarray(tx_inputs["pair_rx"][tx_inputs["valid_true_pairs"]], dtype=int)
        hat_pair_rx = np.asarray(tx_inputs["pair_rx"], dtype=int)

        b_true_term_count_by_tx[tx_idx] = int(true_pair_rx.size)
        b_hat_term_count_by_tx[tx_idx] = int(tx_inputs["h_hat"].shape[0])
        b_true_ntn_count_by_tx[tx_idx] = int(np.unique(true_pair_rx).size) if true_pair_rx.size > 0 else 0
        b_hat_ntn_count_by_tx[tx_idx] = int(np.unique(hat_pair_rx).size) if hat_pair_rx.size > 0 else 0

    return {
        "b_true_ntn_count_by_tx": b_true_ntn_count_by_tx,
        "b_true_term_count_by_tx": b_true_term_count_by_tx,
        "b_hat_ntn_count_by_tx": b_hat_ntn_count_by_tx,
        "b_hat_term_count_by_tx": b_hat_term_count_by_tx,
    }


def run_small_round(
    h_ntn_all: np.ndarray,
    *,
    pairs_by_tx: Dict[int, List[Dict[str, Any]]],
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    round_idx: int,
    lambda_ranges: Iterable[float],
    tx_power: float,
    snr_noise_power: float,
    inr_noise_power: float,
    max_detected_b_terms: str | int | None = "all",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Run one small simulation round where every TX serves one paired TN."""
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx, num_tx_ant).")

    num_ntn_rx = int(h_ntn.shape[0])
    num_tx_total = int(h_ntn.shape[2])
    lambda_list = [float(v) for v in lambda_ranges]

    raw_snr_db: List[float] = []
    perfect_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    est_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    raw_inr_power = np.zeros((num_ntn_rx,), dtype=np.float64)
    perfect_inr_power = {lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list}
    est_inr_power = {lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list}
    eval_mask = np.zeros((num_ntn_rx,), dtype=bool)

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
        raw_snr_db.append(float(pair["snr_raw_db"]))

        h_ntn_tx = np.asarray(h_ntn[:, :, tx_idx, :], dtype=np.complex128)
        raw_inr_power += _interference_power_per_rx(h_ntn_tx, w_t)

        lookup = music_lookup.get(int(tx_idx), {})

        tx_inputs = _extract_tx_b_inputs(
            h_ntn_tx,
            lookup,
            num_tx_ant=h_tn.shape[0],
            max_detected_b_terms=max_detected_b_terms,
        )
        selected_rx_t = np.asarray(tx_inputs["selected_rx"], dtype=int)
        if selected_rx_t.size > 0:
            eval_mask[selected_rx_t] = True
        h_true_t = np.asarray(tx_inputs["h_true"], dtype=np.complex128)
        h_hat_t = np.asarray(tx_inputs["h_hat"], dtype=np.complex128)

        interference_term_true = _covariance_from_channel_vectors(
            h_true_t,
            num_tx_ant=h_tn.shape[0],
        )
        interference_term_hat = _covariance_from_channel_vectors(
            h_hat_t,
            num_tx_ant=h_tn.shape[0],
        )

        for lambda_ in lambda_list:
            v_null_true, _, _, _ = nulling_bf(h_tn, w_r, interference_term_true, lambda_)
            v_null_hat, _, _, _ = nulling_bf(h_tn, w_r, interference_term_hat, lambda_)

            perfect_snr_linear = (
                np.abs((v_null_true.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )
            est_snr_linear = (
                np.abs((v_null_hat.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )
            perfect_snr_db[lambda_].append(float(_safe_db(perfect_snr_linear, eps=eps)))
            est_snr_db[lambda_].append(float(_safe_db(est_snr_linear, eps=eps)))
            perfect_inr_power[lambda_] += _interference_power_per_rx(h_ntn_tx, v_null_true)
            est_inr_power[lambda_] += _interference_power_per_rx(h_ntn_tx, v_null_hat)

    raw_inr_db = (
        _safe_db(raw_inr_power[eval_mask] * float(tx_power) / float(inr_noise_power), eps=eps)
        if np.any(eval_mask)
        else np.empty((0,), dtype=np.float64)
    )
    perfect_inr_db = {
        lambda_: (
            _safe_db(
                perfect_inr_power[lambda_][eval_mask] * float(tx_power) / float(inr_noise_power),
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
                est_inr_power[lambda_][eval_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(eval_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list
    }

    return {
        "raw_snr_db": np.asarray(raw_snr_db, dtype=np.float64),
        "perfect_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in perfect_snr_db.items()},
        "est_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_db.items()},
        "null_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_db.items()},
        "raw_inr_db": np.asarray(raw_inr_db, dtype=np.float64),
        "perfect_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in perfect_inr_db.items()},
        "est_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_db.items()},
        "null_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_db.items()},
        "detected_mask": eval_mask,
        "detected_count": int(np.count_nonzero(eval_mask)),
    }


def run_nulling_cdf_experiment(
    scene_config: Any,
    *,
    num_macro_sims: int,
    compute_positions_kwargs: Dict[str, Any],
    compute_paths_kwargs: Dict[str, Any],
    lambda_ranges: Iterable[float],
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
    resample_satellite_per_macro: bool = False,
    satellite_azimuth_range_deg: Tuple[float, float] = (0.0, 360.0),
    satellite_elevation_range_deg: Tuple[float, float] = (35.0, 90.0),
    satellite_rng_seed: int | None = None,
    max_detected_b_terms: str | int | None = "all",
) -> Dict[str, Any]:
    """Run n macro simulations and m min-count small rounds per macro simulation."""
    if int(num_macro_sims) <= 0:
        raise ValueError("num_macro_sims must be positive.")
    resolved_b_term_limit = _resolve_b_term_limit(max_detected_b_terms)

    try:
        from tqdm.auto import trange
    except Exception:
        trange = None

    lambda_list = [float(v) for v in lambda_ranges]
    raw_snr_all: List[float] = []
    raw_inr_all: List[float] = []
    perfect_snr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    est_snr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    perfect_inr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    est_inr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
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
        b_usage_by_tx = summarize_b_usage_by_tx(
            h_ntn_all,
            music_lookup=music_lookup,
            max_detected_b_terms=resolved_b_term_limit,
        )

        detected_rx_union = np.asarray(ntn_music_out.get("detected_rx_indices_unique", []), dtype=int)
        interfered_ntn_count = int(np.count_nonzero(np.any(np.abs(h_ntn_all) > eps, axis=(1, 2, 3))))
        pair_counts_by_tx = np.asarray(pairing["pair_counts_by_tx"], dtype=int)

        macro_stats.append(
            {
                "sim_idx": int(sim_idx),
                "min_count": int(min_count),
                "pair_counts_by_tx": pair_counts_by_tx.copy(),
                "detected_ntn_count": int(detected_rx_union.size),
                "interfered_ntn_count": interfered_ntn_count,
                "b_true_ntn_count_by_tx": np.asarray(
                    b_usage_by_tx["b_true_ntn_count_by_tx"],
                    dtype=int,
                ),
                "b_true_term_count_by_tx": np.asarray(
                    b_usage_by_tx["b_true_term_count_by_tx"],
                    dtype=int,
                ),
                "b_hat_ntn_count_by_tx": np.asarray(
                    b_usage_by_tx["b_hat_ntn_count_by_tx"],
                    dtype=int,
                ),
                "b_hat_term_count_by_tx": np.asarray(
                    b_usage_by_tx["b_hat_term_count_by_tx"],
                    dtype=int,
                ),
                "satellite_azimuth_deg": float(sat_azimuth_deg),
                "satellite_elevation_deg": float(sat_elevation_deg),
                "satellite_look_pos": sat_look_pos.copy(),
                "angle_metrics": music_quality["angle_metrics"],
                "detected_subset_metrics": music_quality["detected_subset_metrics"],
                "detected_pairs_summary": music_quality["detected_pairs_summary"],
            }
        )

        if min_count <= 0:
            continue

        for round_idx in range(min_count):
            round_out = run_small_round(
                h_ntn_all,
                pairs_by_tx=pairing["pairs_by_tx"],
                music_lookup=music_lookup,
                round_idx=int(round_idx),
                lambda_ranges=lambda_list,
                tx_power=float(tx_power),
                snr_noise_power=float(snr_noise_power),
                inr_noise_power=float(inr_noise_power),
                max_detected_b_terms=resolved_b_term_limit,
                eps=eps,
            )

            raw_snr_all.extend(np.asarray(round_out["raw_snr_db"], dtype=np.float64).tolist())
            raw_inr_all.extend(np.asarray(round_out["raw_inr_db"], dtype=np.float64).tolist())
            for lambda_ in lambda_list:
                perfect_snr_all[lambda_].extend(
                    np.asarray(round_out["perfect_snr_db"][lambda_], dtype=np.float64).tolist()
                )
                est_snr_all[lambda_].extend(
                    np.asarray(round_out["est_snr_db"][lambda_], dtype=np.float64).tolist()
                )
                perfect_inr_all[lambda_].extend(
                    np.asarray(round_out["perfect_inr_db"][lambda_], dtype=np.float64).tolist()
                )
                est_inr_all[lambda_].extend(
                    np.asarray(round_out["est_inr_db"][lambda_], dtype=np.float64).tolist()
                )

    return {
        "raw_snr_db": np.asarray(raw_snr_all, dtype=np.float64),
        "raw_inr_db": np.asarray(raw_inr_all, dtype=np.float64),
        "perfect_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in perfect_snr_all.items()},
        "est_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_all.items()},
        "null_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_snr_all.items()},
        "perfect_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in perfect_inr_all.items()},
        "est_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_all.items()},
        "null_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in est_inr_all.items()},
        "macro_stats": macro_stats,
        "bs_pos_ref": bs_pos_ref,
        "lambda_ranges": np.asarray(lambda_list, dtype=np.float64),
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
        "raw_inr_db": np.asarray(experiment_out["raw_inr_db"], dtype=np.float64),
        "lambda_ranges": np.asarray(experiment_out["lambda_ranges"], dtype=np.float64),
        "max_detected_b_terms": np.asarray([str(experiment_out.get("max_detected_b_terms", "all"))]),
    }
    if experiment_out.get("bs_pos_ref") is not None:
        save_dict["bs_pos_ref"] = np.asarray(experiment_out["bs_pos_ref"], dtype=np.float64)
    for lambda_, vals in experiment_out["perfect_snr_db"].items():
        save_dict[f"perfect_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["est_snr_db"].items():
        save_dict[f"est_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["null_snr_db"].items():
        save_dict[f"null_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["perfect_inr_db"].items():
        save_dict[f"perfect_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["est_inr_db"].items():
        save_dict[f"est_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["null_inr_db"].items():
        save_dict[f"null_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)

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
            [np.asarray(row.get("satellite_look_pos", [np.nan, np.nan, np.nan]), dtype=np.float64) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_pair_counts_by_tx"] = np.stack(
            [np.asarray(row["pair_counts_by_tx"], dtype=int) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_b_true_ntn_count_by_tx"] = np.stack(
            [np.asarray(row.get("b_true_ntn_count_by_tx", []), dtype=int) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_b_true_term_count_by_tx"] = np.stack(
            [np.asarray(row.get("b_true_term_count_by_tx", []), dtype=int) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_b_hat_ntn_count_by_tx"] = np.stack(
            [np.asarray(row.get("b_hat_ntn_count_by_tx", []), dtype=int) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_b_hat_term_count_by_tx"] = np.stack(
            [np.asarray(row.get("b_hat_term_count_by_tx", []), dtype=int) for row in macro_stats],
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

    np.savez(save_path, **save_dict)
    return save_path
