"""MUSIC-based narrowband detection utilities for NTN interference.

This module is designed for the per-BS channel tensor:
    hi.shape == (num_ntn, num_ntn_ant, num_bs_ant)

Typical usage in your notebook:
    from ntn_music_detection import detect_ntn_music_from_hi
    out = detect_ntn_music_from_hi(hi=h_i, num_sources=None, threshold=3.0)
    mask = ~out["detected_mask_user"]   # keep old semantics if needed
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np


def _as_complex_array(x: np.ndarray) -> np.ndarray:
    """Convert input to a complex ndarray without modifying the original."""
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    return arr.astype(np.complex128) + 0j


def collapse_cir_to_narrowband(cir: np.ndarray) -> np.ndarray:
    """Collapse CIR to narrowband channel tensor with stable axis order.

    Expected CIR axis order from Sionna:
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    The function sums over all trailing axes after tx-ant, returning:
        h_all.shape == (num_rx, num_rx_ant, num_tx, num_tx_ant)
    """
    h = _as_complex_array(cir)
    if h.ndim < 4:
        raise ValueError(
            "cir must have at least 4 dims: "
            "[num_rx, num_rx_ant, num_tx, num_tx_ant, ...]."
        )
    if h.ndim == 4:
        return h
    sum_axes = tuple(range(4, h.ndim))
    return np.sum(h, axis=sum_axes)


def extract_hi_for_tx(
    h_all: np.ndarray,
    tx_index: int,
    *,
    nonzero_only: bool = False,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one TX tensor hi from h_all.

    Parameters
    ----------
    h_all : np.ndarray
        Shape (num_rx, num_rx_ant, num_tx, num_tx_ant).
    tx_index : int
        Flat TX index in [0, num_tx-1].
    nonzero_only : bool
        If True, keep only RX rows with any non-zero channel in hi.
    eps : float
        Threshold for non-zero check.

    Returns
    -------
    hi : np.ndarray
        Shape (num_rx_kept, num_rx_ant, num_tx_ant).
    rx_mask : np.ndarray
        Bool mask over original RX index, shape (num_rx,).
    """
    h = _as_complex_array(h_all)
    if h.ndim != 4:
        raise ValueError(
            "h_all must have shape (num_rx, num_rx_ant, num_tx, num_tx_ant)."
        )

    num_rx, _num_rx_ant, num_tx, _num_tx_ant = h.shape
    t = int(tx_index)
    if t < 0 or t >= num_tx:
        raise ValueError(f"tx_index={t} out of range for num_tx={num_tx}.")

    hi = h[:, :, t, :]
    if not nonzero_only:
        return hi, np.ones((num_rx,), dtype=bool)

    rx_mask = np.any(np.abs(hi) > float(eps), axis=(1, 2))
    return hi[rx_mask], rx_mask


def extract_tx_channel_matrix(
    h_all: np.ndarray,
    tx_index: int,
    *,
    rx_ant_index: int = 0,
    nonzero_only: bool = False,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-TX matrix H_t for one RX antenna.

    Returns H_t with shape (num_rx_kept, num_tx_ant), matching your
    per-sector MUSIC input requirement like (100, 64).
    """
    hi, rx_mask = extract_hi_for_tx(h_all, tx_index, nonzero_only=nonzero_only, eps=eps)
    num_rx_ant = hi.shape[1]
    r = int(rx_ant_index)
    if r < 0 or r >= num_rx_ant:
        raise ValueError(f"rx_ant_index={r} out of range for num_rx_ant={num_rx_ant}.")
    return hi[:, r, :], rx_mask


def _broadcast_powers(
    user_powers: Optional[np.ndarray],
    num_ntn: int,
    num_ntn_ant: int,
) -> np.ndarray:
    """Broadcast user powers to shape (num_ntn, num_ntn_ant)."""
    if user_powers is None:
        return np.ones((num_ntn, num_ntn_ant), dtype=np.float64)

    p = np.asarray(user_powers, dtype=np.float64)
    if p.ndim == 1:
        if p.shape[0] != num_ntn:
            raise ValueError(
                "user_powers has shape (num_ntn,), but num_ntn does not match."
            )
        return np.repeat(p[:, None], num_ntn_ant, axis=1)

    if p.ndim == 2 and p.shape == (num_ntn, num_ntn_ant):
        return p

    raise ValueError(
        "user_powers must be None, shape (num_ntn,), or shape (num_ntn, num_ntn_ant)."
    )


def _covariance_from_static_channels(
    hi: np.ndarray,
    user_powers_2d: np.ndarray,
    noise_var: float,
) -> np.ndarray:
    """Build covariance analytically for static narrowband channels.

    Model:
        x = sum_{u,r} sqrt(p_{u,r}) h_{u,r} s_{u,r} + w
        E[s s^H] = I, E[w w^H] = noise_var * I
    """
    num_ntn, num_ntn_ant, num_bs_ant = hi.shape
    rxx = np.zeros((num_bs_ant, num_bs_ant), dtype=np.complex128)

    for u in range(num_ntn):
        for r in range(num_ntn_ant):
            h = hi[u, r, :].reshape(-1, 1)
            rxx += user_powers_2d[u, r] * (h @ h.conj().T)

    if noise_var > 0.0:
        rxx += noise_var * np.eye(num_bs_ant, dtype=np.complex128)
    return rxx


def _sample_covariance_from_snapshots(
    hi: np.ndarray,
    user_powers_2d: np.ndarray,
    noise_var: float,
    num_snapshots: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic snapshots and return (Rxx, X).

    - No QAM order is required here; symbols are circular Gaussian CN(0, 1).
    - This is standard for subspace estimation and is modulation-agnostic.
    """
    num_ntn, num_ntn_ant, num_bs_ant = hi.shape
    ur = num_ntn * num_ntn_ant

    # H shape: (M, UR), each column is one source channel vector to BS array.
    h_mat = hi.reshape(ur, num_bs_ant).T
    p_vec = user_powers_2d.reshape(ur)
    p_sqrt = np.sqrt(np.maximum(p_vec, 0.0))

    s = (
        rng.standard_normal((ur, num_snapshots))
        + 1j * rng.standard_normal((ur, num_snapshots))
    ) / np.sqrt(2.0)
    s *= p_sqrt[:, None]

    x_clean = h_mat @ s
    if noise_var > 0.0:
        w = (
            rng.standard_normal((num_bs_ant, num_snapshots))
            + 1j * rng.standard_normal((num_bs_ant, num_snapshots))
        ) * np.sqrt(noise_var / 2.0)
        x = x_clean + w
    else:
        x = x_clean

    rxx = (x @ x.conj().T) / float(num_snapshots)
    return rxx, x


def _estimate_num_sources_mdl(
    eigenvalues_desc: np.ndarray,
    num_snapshots: int,
    max_sources: Optional[int] = None,
) -> int:
    """Estimate source count with Wax-Kailath MDL."""
    eig = np.real(np.asarray(eigenvalues_desc, dtype=np.float64))
    m = eig.shape[0]
    if max_sources is None:
        max_sources = m - 1
    max_sources = int(np.clip(max_sources, 0, m - 1))

    # MDL is meaningful with at least a few snapshots.
    n = int(max(num_snapshots, m + 1))
    eps = 1e-12

    mdl_vals = np.full(max_sources + 1, np.inf, dtype=np.float64)
    for k in range(max_sources + 1):
        noise_eigs = np.maximum(eig[k:], eps)
        p = m - k
        if p <= 0:
            continue
        gm = np.exp(np.mean(np.log(noise_eigs)))
        am = np.mean(noise_eigs)
        if am <= eps:
            continue
        # Wax-Kailath MDL:
        # MDL(k) = -n*(m-k)*log(gm/am) + 0.5*k*(2m-k)*log(n)
        mdl_vals[k] = -n * p * np.log(gm / am) + 0.5 * k * (2 * m - k) * np.log(n)

    return int(np.argmin(mdl_vals))


def _estimate_num_sources_energy(
    eigenvalues_desc: np.ndarray,
    energy_ratio: float = 0.95,
) -> int:
    """Fallback source-count estimate by cumulative eigen-energy."""
    eig = np.real(np.asarray(eigenvalues_desc, dtype=np.float64))
    eig = np.maximum(eig, 0.0)
    total = float(np.sum(eig))
    if total <= 0.0:
        return 0
    csum = np.cumsum(eig) / total
    k = int(np.searchsorted(csum, energy_ratio) + 1)
    return int(np.clip(k, 0, eig.shape[0] - 1))


def _compute_user_scores(
    hi: np.ndarray,
    us: np.ndarray,
    en: np.ndarray,
    reduce_ntn_ant: Literal["max", "mean"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MUSIC-like user scores from signal/noise subspace projections.

    score = ||Us^H h||^2 / ||En^H h||^2 for normalized h.
    """
    num_ntn, num_ntn_ant, _ = hi.shape
    eps = 1e-12
    score_per_ant = np.zeros((num_ntn, num_ntn_ant), dtype=np.float64)

    for u in range(num_ntn):
        for r in range(num_ntn_ant):
            h = hi[u, r, :].reshape(-1, 1)
            norm_h = np.linalg.norm(h)
            if norm_h <= eps:
                score_per_ant[u, r] = 0.0
                continue
            h_n = h / norm_h

            sig_proj = np.linalg.norm(us.conj().T @ h_n) ** 2 if us.size else 0.0
            noi_proj = np.linalg.norm(en.conj().T @ h_n) ** 2 if en.size else eps
            score_per_ant[u, r] = float(sig_proj / max(noi_proj, eps))

    if reduce_ntn_ant == "max":
        score_user = np.max(score_per_ant, axis=1)
    elif reduce_ntn_ant == "mean":
        score_user = np.mean(score_per_ant, axis=1)
    else:
        raise ValueError("reduce_ntn_ant must be 'max' or 'mean'.")

    return score_per_ant, score_user


def detect_ntn_music_from_hi(
    hi: np.ndarray,
    *,
    num_sources: Optional[int] = None,
    threshold: Optional[float] = 1.0,
    user_powers: Optional[np.ndarray] = None,
    noise_var: float = 0.0,
    covariance_mode: Literal["analytic", "sample"] = "analytic",
    num_snapshots: int = 200,
    rng_seed: Optional[int] = None,
    source_estimation: Literal["mdl", "energy"] = "mdl",
    energy_ratio: float = 0.95,
    reduce_ntn_ant: Literal["max", "mean"] = "max",
) -> Dict[str, np.ndarray]:
    """Run narrowband MUSIC detection for one BS/sector channel tensor.

    Parameters
    ----------
    hi : np.ndarray
        Shape (num_ntn, num_ntn_ant, num_bs_ant), complex channel tensor.
    num_sources : int | None
        Signal subspace dimension K. If None, estimated automatically.
    threshold : float | None
        Detection threshold on user score.
        - If provided: detected_mask_user = score_user >= threshold
        - If None: top-K users are marked as detected.
    user_powers : np.ndarray | None
        Optional source powers:
        - shape (num_ntn,)
        - or shape (num_ntn, num_ntn_ant)
    noise_var : float
        Noise variance per BS antenna.
    covariance_mode : {"analytic", "sample"}
        "analytic": Rxx from static channels directly.
        "sample": synthetic snapshots are generated and sample covariance is used.
    num_snapshots : int
        Number of snapshots when covariance_mode == "sample".
    rng_seed : int | None
        Random seed for reproducibility in sample mode.
    source_estimation : {"mdl", "energy"}
        Automatic K-estimation rule when num_sources is None.
    energy_ratio : float
        Used only when source_estimation == "energy".
    reduce_ntn_ant : {"max", "mean"}
        How to aggregate multi-antenna NTN scores to user-level score.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys:
        - "detected_mask_user": bool array, shape (num_ntn,)
        - "detected_mask_per_ant": bool array, shape (num_ntn, num_ntn_ant)
        - "score_user": float array, shape (num_ntn,)
        - "score_per_ant": float array, shape (num_ntn, num_ntn_ant)
        - "num_sources_est": int scalar in ndarray
        - "threshold_used": float scalar in ndarray
        - "eigenvalues_desc": float array, shape (num_bs_ant,)
        - "covariance": complex array, shape (num_bs_ant, num_bs_ant)
    """
    hi_c = _as_complex_array(hi)
    if hi_c.ndim != 3:
        raise ValueError(
            "hi must be a 3D tensor with shape (num_ntn, num_ntn_ant, num_bs_ant)."
        )

    num_ntn, num_ntn_ant, num_bs_ant = hi_c.shape
    if num_bs_ant < 2:
        raise ValueError("MUSIC requires num_bs_ant >= 2.")
    if noise_var < 0.0:
        raise ValueError("noise_var must be >= 0.")
    if covariance_mode == "sample" and num_snapshots < 2:
        raise ValueError("num_snapshots must be >= 2 in sample mode.")

    p_2d = _broadcast_powers(user_powers, num_ntn=num_ntn, num_ntn_ant=num_ntn_ant)

    if covariance_mode == "analytic":
        rxx = _covariance_from_static_channels(hi_c, p_2d, noise_var)
        n_for_mdl = max(num_ntn * num_ntn_ant, num_bs_ant + 1)
    elif covariance_mode == "sample":
        rng = np.random.default_rng(rng_seed)
        rxx, _x = _sample_covariance_from_snapshots(
            hi=hi_c,
            user_powers_2d=p_2d,
            noise_var=noise_var,
            num_snapshots=num_snapshots,
            rng=rng,
        )
        n_for_mdl = int(num_snapshots)
    else:
        raise ValueError("covariance_mode must be 'analytic' or 'sample'.")

    # Hermitian eigendecomposition.
    evals, evecs = np.linalg.eigh(rxx)
    idx = np.argsort(np.real(evals))[::-1]
    evals_desc = np.real(evals[idx])
    evecs_desc = evecs[:, idx]

    if num_sources is None:
        if source_estimation == "mdl":
            k_est = _estimate_num_sources_mdl(
                eigenvalues_desc=evals_desc, num_snapshots=n_for_mdl
            )
        elif source_estimation == "energy":
            k_est = _estimate_num_sources_energy(
                eigenvalues_desc=evals_desc, energy_ratio=energy_ratio
            )
        else:
            raise ValueError("source_estimation must be 'mdl' or 'energy'.")
    else:
        k_est = int(num_sources)

    k_est = int(np.clip(k_est, 0, num_bs_ant - 1))
    us = evecs_desc[:, :k_est] if k_est > 0 else np.empty((num_bs_ant, 0), dtype=np.complex128)
    en = evecs_desc[:, k_est:] if k_est < num_bs_ant else np.empty((num_bs_ant, 0), dtype=np.complex128)

    score_per_ant, score_user = _compute_user_scores(
        hi=hi_c,
        us=us,
        en=en,
        reduce_ntn_ant=reduce_ntn_ant,
    )

    if threshold is None:
        # If no threshold is provided, mark top-K users as detected.
        k_users = int(np.clip(k_est, 0, num_ntn))
        detected = np.zeros(num_ntn, dtype=bool)
        if k_users > 0:
            top_idx = np.argsort(score_user)[::-1][:k_users]
            detected[top_idx] = True
        detected_per_ant = np.repeat(detected[:, None], num_ntn_ant, axis=1)
        threshold_used = np.nan
    else:
        threshold_used = float(threshold)
        detected = score_user >= threshold_used
        detected_per_ant = score_per_ant >= threshold_used

    return {
        "detected_mask_user": detected,
        "detected_mask_per_ant": detected_per_ant,
        "score_user": score_user,
        "score_per_ant": score_per_ant,
        "num_sources_est": np.array(k_est, dtype=np.int64),
        "threshold_used": np.array(threshold_used, dtype=np.float64),
        "eigenvalues_desc": evals_desc,
        "covariance": rxx,
    }


def _rotation_matrix_xyz(
    yaw: float,
    pitch: float,
    roll: float,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> np.ndarray:
    """Create local->global rotation matrix from yaw/pitch/roll (radians).

    We interpret:
    - yaw   : rotation around global z-axis
    - pitch : rotation around local/global y-axis (per order)
    - roll  : rotation around local/global x-axis (per order)
    """
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

    mats = {"x": rx, "y": ry, "z": rz}
    if len(rotation_order) != 3 or any(ax not in mats for ax in rotation_order):
        raise ValueError("rotation_order must be a permutation of 'x', 'y', 'z' (e.g., 'zyx').")

    r = np.eye(3, dtype=np.float64)
    for ax in rotation_order:
        r = r @ mats[ax]
    return r


def _unit_vector_from_angles(phi_deg: float, theta_deg: float) -> np.ndarray:
    """Global unit vector from spherical angles (phi azimuth, theta zenith)."""
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )


def _boresight_global_from_orientation(
    orientation_rad: Tuple[float, float, float],
    *,
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> np.ndarray:
    """Return panel boresight (local normal of panel plane) in global coordinates.

    Panel normal in local coordinates:
    - "yz" -> +x
    - "xz" -> +y
    - "xy" -> +z
    """
    yaw, pitch, roll = orientation_rad
    r_local_to_global = _rotation_matrix_xyz(yaw, pitch, roll, rotation_order=rotation_order)
    if panel_plane == "yz":
        local_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif panel_plane == "xz":
        local_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif panel_plane == "xy":
        local_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError("panel_plane must be 'yz', 'xz', or 'xy'.")
    return r_local_to_global @ local_axis


def upa_steering_global(
    phi_deg: float,
    theta_deg: float,
    num_rows: int,
    num_cols: int,
    *,
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    d_h: float = 0.5,
    d_v: float = 0.5,
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    phase_sign: Literal[-1, 1] = 1,
    flatten_order: Literal["C", "F"] = "C",
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> np.ndarray:
    """UPA steering with global angles and panel orientation.

    Inputs phi/theta are interpreted as global spherical angles (degrees):
    - phi   : azimuth, in x-y plane from +x
    - theta : zenith, from +z (0..180)
    - panel_plane:
      - "yz": horizontal axis is local y, vertical axis is local z (default)
      - "xz": horizontal axis is local x, vertical axis is local z
      - "xy": horizontal axis is local x, vertical axis is local y
    - phase_sign:
      - +1 or -1 for convention matching with channel phasor sign.
    """
    u_global = _unit_vector_from_angles(phi_deg, theta_deg)

    yaw, pitch, roll = orientation_rad
    r_local_to_global = _rotation_matrix_xyz(yaw, pitch, roll, rotation_order=rotation_order)

    # Convert global direction to local panel coordinates.
    u_local = r_local_to_global.T @ u_global
    if panel_plane == "yz":
        u_h = float(u_local[1])
        u_v = float(u_local[2])
    elif panel_plane == "xz":
        u_h = float(u_local[0])
        u_v = float(u_local[2])
    elif panel_plane == "xy":
        u_h = float(u_local[0])
        u_v = float(u_local[1])
    else:
        raise ValueError("panel_plane must be 'yz', 'xz', or 'xy'.")

    rows = np.arange(num_rows, dtype=np.float64)
    cols = np.arange(num_cols, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    sgn = 1.0 if int(phase_sign) >= 0 else -1.0
    if flatten_order not in ("C", "F"):
        raise ValueError("flatten_order must be 'C' or 'F'.")

    phase = sgn * 2.0 * np.pi * (d_h * cc * u_h + d_v * rr * u_v)
    a = np.exp(1j * phase).reshape(-1, 1, order=flatten_order)
    return a / np.sqrt(float(num_rows * num_cols))


def noise_subspace_from_music_out(music_out: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract noise subspace En from MUSIC output dict."""
    rxx = np.asarray(music_out["covariance"])
    k_est = int(np.asarray(music_out["num_sources_est"]).item())
    m = int(rxx.shape[0])
    k_est = int(np.clip(k_est, 0, m - 1))

    evals, evecs = np.linalg.eigh(rxx)
    idx = np.argsort(np.real(evals))[::-1]
    evecs = evecs[:, idx]
    return evecs[:, k_est:] if k_est < m else np.empty((m, 0), dtype=np.complex128)


def music_top_peaks(
    en: np.ndarray,
    *,
    num_rows: int,
    num_cols: int,
    phi_grid_deg: Iterable[float],
    theta_grid_deg: Iterable[float],
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    phase_sign: Literal[-1, 1] = 1,
    flatten_order: Literal["C", "F"] = "C",
    forward_only: bool = False,
    forward_cos_min: float = 0.0,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
    top_n: int = 24,
    min_sep_phi_deg: float = 4.0,
    min_sep_theta_deg: float = 4.0,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """Find dominant MUSIC pseudo-spectrum peaks.

    Returns list of tuples:
        (pseudo_power, phi_deg, theta_deg, steering_vec)
    """
    if en.size == 0:
        return []

    g = en @ en.conj().T
    boresight_global = _boresight_global_from_orientation(
        orientation_rad=orientation_rad,
        panel_plane=panel_plane,
        rotation_order=rotation_order,
    )
    fwd_cos = float(np.clip(forward_cos_min, -1.0, 1.0))

    cand: List[Tuple[float, float, float, np.ndarray]] = []
    for th in theta_grid_deg:
        for ph in phi_grid_deg:
            if forward_only:
                u_global = _unit_vector_from_angles(float(ph), float(th))
                if float(np.dot(u_global, boresight_global)) < fwd_cos:
                    continue

            a = upa_steering_global(
                float(ph),
                float(th),
                num_rows,
                num_cols,
                orientation_rad=orientation_rad,
                panel_plane=panel_plane,
                phase_sign=phase_sign,
                flatten_order=flatten_order,
                rotation_order=rotation_order,
            )
            den = np.real((a.conj().T @ g @ a).item())
            p = 1.0 / max(float(den), 1e-12)
            cand.append((p, float(ph), float(th), a))

    cand.sort(key=lambda x: x[0], reverse=True)

    peaks: List[Tuple[float, float, float, np.ndarray]] = []
    for p, ph, th, a in cand:
        keep = True
        for _, sph, sth, _ in peaks:
            dphi = abs(((ph - sph + 180.0) % 360.0) - 180.0)
            dth = abs(th - sth)
            if dphi < min_sep_phi_deg and dth < min_sep_theta_deg:
                keep = False
                break
        if keep:
            peaks.append((p, ph, th, a))
            if len(peaks) >= int(top_n):
                break
    return peaks


def assign_hat_angle_to_user(
    h_user_vec: np.ndarray,
    peaks: List[Tuple[float, float, float, np.ndarray]],
) -> Tuple[float, float]:
    """Assign one (phi,theta) estimate to one user channel vector by best steering match."""
    if len(peaks) == 0:
        return float("nan"), float("nan")

    h = _as_complex_array(h_user_vec).reshape(-1, 1)
    nh = float(np.linalg.norm(h))
    if nh <= 1e-12:
        return float("nan"), float("nan")
    h = h / nh

    best_idx = 0
    best_val = -1.0
    for i, (_p, ph, th, a) in enumerate(peaks):
        val = float(np.abs((a.conj().T @ h).item()))
        if val > best_val:
            best_val = val
            best_idx = i

    _p, ph, th, _a = peaks[best_idx]
    return float(ph), float(th)


def build_steering_bank(
    *,
    num_rows: int,
    num_cols: int,
    phi_grid_deg: Iterable[float],
    theta_grid_deg: Iterable[float],
    orientation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    panel_plane: Literal["yz", "xz", "xy"] = "yz",
    phase_sign: Literal[-1, 1] = 1,
    flatten_order: Literal["C", "F"] = "C",
    forward_only: bool = False,
    forward_cos_min: float = 0.0,
    rotation_order: Literal["zyx", "zxy", "yxz", "yzx", "xyz", "xzy"] = "zyx",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build steering dictionary A and corresponding angle arrays.

    Returns
    -------
    A : np.ndarray
        Shape (M, G), M=num_rows*num_cols.
    phi_bank_deg : np.ndarray
        Shape (G,).
    theta_bank_deg : np.ndarray
        Shape (G,).
    """
    boresight_global = _boresight_global_from_orientation(
        orientation_rad=orientation_rad,
        panel_plane=panel_plane,
        rotation_order=rotation_order,
    )
    fwd_cos = float(np.clip(forward_cos_min, -1.0, 1.0))

    vecs: List[np.ndarray] = []
    phi_list: List[float] = []
    theta_list: List[float] = []

    for th in theta_grid_deg:
        for ph in phi_grid_deg:
            phf = float(ph)
            thf = float(th)
            if forward_only:
                u_global = _unit_vector_from_angles(phf, thf)
                if float(np.dot(u_global, boresight_global)) < fwd_cos:
                    continue

            a = upa_steering_global(
                phf,
                thf,
                num_rows,
                num_cols,
                orientation_rad=orientation_rad,
                panel_plane=panel_plane,
                phase_sign=phase_sign,
                flatten_order=flatten_order,
                rotation_order=rotation_order,
            )
            vecs.append(a)
            phi_list.append(phf)
            theta_list.append(thf)

    m = int(num_rows * num_cols)
    if len(vecs) == 0:
        return (
            np.empty((m, 0), dtype=np.complex128),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    a_bank = np.hstack(vecs)
    phi_bank = np.asarray(phi_list, dtype=np.float64)
    theta_bank = np.asarray(theta_list, dtype=np.float64)
    return a_bank, phi_bank, theta_bank


def estimate_angle_from_channel_scan(
    h_user_vec: np.ndarray,
    a_bank: np.ndarray,
    phi_bank_deg: np.ndarray,
    theta_bank_deg: np.ndarray,
    *,
    scan_mode: Literal["complex", "phase_only"] = "complex",
) -> Tuple[float, float, float]:
    """Estimate one user angle by scanning steering dictionary against channel vector.

    Score is |a^H h| with normalized h and normalized steering columns.
    Returns (phi_hat_deg, theta_hat_deg, score_max).
    """
    if a_bank.size == 0 or phi_bank_deg.size == 0:
        return float("nan"), float("nan"), float("nan")

    a = _as_complex_array(a_bank)
    if a.ndim != 2:
        raise ValueError("a_bank must be a 2D matrix with shape (num_ant, num_grid).")

    h = _as_complex_array(h_user_vec).reshape(-1, 1)
    if a.shape[0] != h.shape[0]:
        raise ValueError(
            f"Dimension mismatch: steering has {a.shape[0]} antennas, "
            f"but h_user_vec has {h.shape[0]} elements."
        )
    if a.shape[1] != int(np.asarray(phi_bank_deg).size) or a.shape[1] != int(np.asarray(theta_bank_deg).size):
        raise ValueError("a_bank column count must match phi_bank_deg/theta_bank_deg length.")

    nh = float(np.linalg.norm(h))
    if nh <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    h = h / nh

    if scan_mode not in ("complex", "phase_only"):
        raise ValueError("scan_mode must be 'complex' or 'phase_only'.")

    if scan_mode == "complex":
        # A columns are normalized by construction.
        corrs = np.abs((a.conj().T @ h).reshape(-1))
    else:
        # Phase-only matching is robust when element-pattern amplitudes distort h.
        a_phase = np.exp(1j * np.angle(a))
        h_phase = np.exp(1j * np.angle(h))
        corrs = np.abs((a_phase.conj().T @ h_phase).reshape(-1))

    if corrs.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = int(np.argmax(corrs))
    return float(phi_bank_deg[idx]), float(theta_bank_deg[idx]), float(corrs[idx])


def aod_to_aoa_reverse(phi_t_deg: np.ndarray, theta_t_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert downlink AOD (TX->RX) to reversed-link AOA reference (RX<-TX)."""
    phi = (np.asarray(phi_t_deg, dtype=np.float64) + 180.0) % 360.0
    theta = 180.0 - np.asarray(theta_t_deg, dtype=np.float64)
    return phi, theta


def zenith_to_elevation_deg(theta_zenith_deg: np.ndarray) -> np.ndarray:
    """Convert zenith angle (0=+z, 90=horizontal) to elevation (0=horizontal)."""
    return 90.0 - np.asarray(theta_zenith_deg, dtype=np.float64)


def elevation_to_zenith_deg(theta_elevation_deg: np.ndarray) -> np.ndarray:
    """Convert elevation angle (0=horizontal) to zenith (0=+z, 90=horizontal)."""
    return 90.0 - np.asarray(theta_elevation_deg, dtype=np.float64)


def sector_index_from_tx_index(tx_index: np.ndarray, nsect: int) -> np.ndarray:
    """Map flat TX index to sector index."""
    if int(nsect) <= 0:
        raise ValueError("nsect must be > 0.")
    return np.asarray(tx_index, dtype=np.int64) % int(nsect)


def sector_yaw_offset_deg(sector_index: np.ndarray, nsect: int) -> np.ndarray:
    """Sector azimuth offsets in degrees for equally-spaced sectors."""
    if int(nsect) <= 0:
        raise ValueError("nsect must be > 0.")
    s = np.asarray(sector_index, dtype=np.float64)
    return (360.0 * s / float(nsect)) % 360.0


def sector_local_aod_to_global(
    phi_local_deg: np.ndarray,
    *,
    sector_index: np.ndarray,
    nsect: int,
    yaw_offset_deg: float = 0.0,
) -> np.ndarray:
    """Convert per-sector local azimuth to global azimuth.

    For 3-sector deployment, this applies +0/+120/+240 deg (plus optional offset).
    """
    phi_loc = np.asarray(phi_local_deg, dtype=np.float64)
    sec = np.asarray(sector_index, dtype=np.float64)
    if phi_loc.shape != sec.shape:
        raise ValueError("phi_local_deg and sector_index must have the same shape.")

    sec_off = sector_yaw_offset_deg(sec, int(nsect))
    return (phi_loc + sec_off + float(yaw_offset_deg)) % 360.0


def sionna_aod_to_uplink_aoa(
    phi_t_deg: np.ndarray,
    theta_t_deg: np.ndarray,
    *,
    phi_is_sector_local: bool = False,
    sector_index: Optional[np.ndarray] = None,
    nsect: Optional[int] = None,
    yaw_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Sionna downlink AOD reference to uplink AOA at BS.

    - If phi_t is sector-local, set phi_is_sector_local=True and provide
      sector_index + nsect to first convert it to global azimuth.
    - Zenith angle conversion is always theta_aoa = 180 - theta_t.
    """
    phi_t = np.asarray(phi_t_deg, dtype=np.float64)
    theta_t = np.asarray(theta_t_deg, dtype=np.float64)
    if phi_t.shape != theta_t.shape:
        raise ValueError("phi_t_deg and theta_t_deg must have the same shape.")

    if phi_is_sector_local:
        if sector_index is None or nsect is None:
            raise ValueError(
                "When phi_is_sector_local=True, sector_index and nsect are required."
            )
        phi_global = sector_local_aod_to_global(
            phi_t,
            sector_index=np.asarray(sector_index),
            nsect=int(nsect),
            yaw_offset_deg=float(yaw_offset_deg),
        )
    else:
        phi_global = phi_t % 360.0

    return aod_to_aoa_reverse(phi_global, theta_t)


def wrapped_angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Wrapped angular error a-b in [-180,180)."""
    return ((np.asarray(a_deg) - np.asarray(b_deg) + 180.0) % 360.0) - 180.0


def angle_error_metrics(
    phi_true_deg: np.ndarray,
    theta_true_deg: np.ndarray,
    phi_hat_deg: np.ndarray,
    theta_hat_deg: np.ndarray,
    *,
    reference_mode: Literal["aod", "aoa_reverse"] = "aoa_reverse",
) -> Dict[str, float]:
    """Compute MAE/STD metrics for angle error under a reference convention."""
    phi_true = np.asarray(phi_true_deg, dtype=np.float64)
    theta_true = np.asarray(theta_true_deg, dtype=np.float64)
    phi_hat = np.asarray(phi_hat_deg, dtype=np.float64)
    theta_hat = np.asarray(theta_hat_deg, dtype=np.float64)

    mask = ~(np.isnan(phi_true) | np.isnan(theta_true) | np.isnan(phi_hat) | np.isnan(theta_hat))
    if np.count_nonzero(mask) == 0:
        return {
            "count": 0.0,
            "phi_mae_deg": float("nan"),
            "phi_std_deg": float("nan"),
            "theta_mae_deg": float("nan"),
            "theta_std_deg": float("nan"),
        }

    phi_true = phi_true[mask]
    theta_true = theta_true[mask]
    phi_hat = phi_hat[mask]
    theta_hat = theta_hat[mask]

    if reference_mode == "aod":
        phi_ref, theta_ref = phi_true, theta_true
    elif reference_mode == "aoa_reverse":
        phi_ref, theta_ref = aod_to_aoa_reverse(phi_true, theta_true)
    else:
        raise ValueError("reference_mode must be 'aod' or 'aoa_reverse'.")

    dphi = wrapped_angle_diff_deg(phi_hat, phi_ref)
    dtheta = theta_hat - theta_ref

    return {
        "count": float(phi_ref.shape[0]),
        "phi_mae_deg": float(np.mean(np.abs(dphi))),
        "phi_std_deg": float(np.std(dphi)),
        "theta_mae_deg": float(np.mean(np.abs(dtheta))),
        "theta_std_deg": float(np.std(dtheta)),
    }
