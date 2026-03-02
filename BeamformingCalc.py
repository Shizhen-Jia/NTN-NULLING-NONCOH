# beamforming_utils.py
import numpy as np

def svd_bf(h: np.ndarray, tx_antennas):
    """
    Performs SVD-based beamforming to extract w_t (transmit beam) and w_r (receive beam).

    Parameters
    ----------
    h : np.ndarray
        The channel matrix, typically of shape (num_tx_antennas, num_rx_antennas).

    Returns
    -------
    w_t : np.ndarray
        The first column of the left singular vectors, of shape (num_tx_antennas, 1).
    w_r : np.ndarray
        The first column of the right singular vectors, of shape (num_rx_antennas, 1).
    """
    
    # h = ((h)*np.sqrt(tx_antennas)  /np.linalg.norm(h))
    h = (h) /np.linalg.norm(h)
    # h = U * diag(s) * Vh
    U, s, Vh = np.linalg.svd(h)
    # Extract the principal singular-vector components
    w_t = U[:, 0].reshape(-1, 1)
    V = Vh.conj().T
    w_r = V[:, 0].reshape(-1, 1)
    return w_t, w_r


def nulling_bf(h: np.ndarray, 
               w_r: np.ndarray, 
               interference_term: np.ndarray, 
               lambda_: float):
    """
    Calculates the nulling vector v_null based on the interference covariance.

    The nulling vector is the principal eigenvector of 
    Q = h * w_r * w_r^H * h^H - lambda_ * interference_term.

    Parameters
    ----------
    h : np.ndarray
        The channel matrix of shape (num_tx_antennas, num_rx_antennas). 
        Must be compatible with w_r (i.e., h.shape[1] == w_r.shape[0]).
    w_t : np.ndarray
        The transmitte beamforming vector, of shape (num_tx_antennas, 1).
    interference_term : np.ndarray
        The aggregated interference covariance matrix, typically shape (num_tx_antennas, num_tx_antennas).
    lambda_ : float
        A weighting factor that balances the desired signal versus the interference penalty.

    Returns
    -------
    v_null : np.ndarray
        The nulling vector, of shape (num_rx_antennas, 1),
        which is used on the transmit side (or receive side, depending on your convention)
        to mitigate interference.
    """
    
    # h= ((h)*np.sqrt(tx_antennas)  /np.linalg.norm(h))
    # h = ((h) /np.linalg.norm(h))
    # interference_term = (interference_term) /np.linalg.norm(interference_term)
    # Build the matrix Q
    A = h @ w_r @ w_r.conj().T @ h.conj().T
    B = lambda_ * interference_term
    Q = A-B
    # Q = h @ w_r @ w_r.conj().T @ h.conj().T - lambda_ * interference_term
    
    # Eigen-decomposition of Q
    eigen_values, v_nulls = np.linalg.eig(Q)

    # Sort eigenvalues from largest to smallest
    idx = np.argsort(eigen_values)[::-1]
    eigen_values_sorted = eigen_values[idx]
    max_eigen_value = eigen_values_sorted[0]
    v_nulls = v_nulls[:, idx]
    # The nulling vector is the eigenvector corresponding to the largest eigenvalue
    v_null = v_nulls[:, 0].reshape(-1, 1)
    
    return v_null, A, B, max_eigen_value


def nulling_bf_music_noncoh(
    h: np.ndarray,
    w_r: np.ndarray,
    u_hat: np.ndarray,
    g_hat: np.ndarray,
    lambda_: float,
    eps: float = 1e-12,
):
    """
    Non-coherent MUSIC-guided nulling:
        Q = h0_tilde h0_tilde^H - lambda * sum_i g_i u_i u_i^H

    where h0 = h @ w_r is the effective desired channel vector.

    Parameters
    ----------
    h : np.ndarray
        Desired TN channel matrix (num_tx_ant, num_rx_ant).
    w_r : np.ndarray
        Receive combiner of the desired TN link (num_rx_ant, 1).
    u_hat : np.ndarray
        MUSIC steering vectors of detected victim NTN users.
        Shape (K, num_tx_ant) or (num_tx_ant,) for a single user.
    g_hat : np.ndarray
        Non-negative scalar weights for each steering vector.
        Typical choice is |alpha_hat|^2. Shape (K,) or scalar.
    lambda_ : float
        Regularization parameter.
    eps : float
        Small number for numerical stability.

    Returns
    -------
    v_null : np.ndarray
        Principal eigenvector of Q, shape (num_tx_ant, 1).
    A : np.ndarray
        Desired-signal term h0_tilde h0_tilde^H.
    B : np.ndarray
        Interference penalty term lambda * sum_i g_i u_i u_i^H.
    max_eigen_value : float
        Largest eigenvalue of Q.
    """
    h = np.asarray(h, dtype=np.complex128)
    w_r = np.asarray(w_r, dtype=np.complex128).reshape(-1, 1)
    num_tx_ant = int(h.shape[0])

    h0 = np.asarray(h @ w_r, dtype=np.complex128).reshape(-1, 1)
    h0_norm = np.linalg.norm(h0)
    if not np.isfinite(h0_norm) or h0_norm <= eps:
        raise ValueError("Desired channel h @ w_r has zero (or invalid) norm.")
    h0_tilde = h0 / (h0_norm + eps)
    A = h0_tilde @ h0_tilde.conj().T

    B_base = np.zeros((num_tx_ant, num_tx_ant), dtype=np.complex128)
    if u_hat is not None and g_hat is not None:
        u_arr = np.asarray(u_hat, dtype=np.complex128)
        if u_arr.ndim == 1:
            u_arr = u_arr.reshape(1, -1)
        if u_arr.shape[1] != num_tx_ant:
            raise ValueError(
                f"u_hat size mismatch: expected {num_tx_ant} antennas, got {u_arr.shape[1]}."
            )

        g_arr = np.asarray(g_hat).reshape(-1)
        if g_arr.size == 1 and u_arr.shape[0] > 1:
            g_arr = np.full((u_arr.shape[0],), float(np.real(g_arr[0])), dtype=float)
        if g_arr.size != u_arr.shape[0]:
            raise ValueError(
                f"g_hat length mismatch: got {g_arr.size}, expected {u_arr.shape[0]}."
            )

        # Keep only finite positive real weights.
        g_arr = np.real(g_arr.astype(np.complex128))
        valid = np.isfinite(g_arr) & (g_arr > 0.0)
        if np.any(valid):
            u_valid = u_arr[valid]
            g_valid = g_arr[valid]
            # Normalize each steering vector before outer-product accumulation.
            u_norm = np.linalg.norm(u_valid, axis=1, keepdims=True)
            u_valid = u_valid / np.maximum(u_norm, eps)
            B_base = np.einsum(
                "k,ki,kj->ij",
                g_valid,
                u_valid,
                np.conjugate(u_valid),
                optimize=True,
            )

    B = float(lambda_) * B_base
    Q = A - B
    # Enforce Hermitian symmetry before eigh to avoid tiny numeric asymmetry.
    Q = 0.5 * (Q + Q.conj().T)

    eigen_values, eigen_vectors = np.linalg.eigh(Q)
    idx = int(np.argmax(eigen_values))
    max_eigen_value = float(np.real(eigen_values[idx]))
    v_null = np.asarray(eigen_vectors[:, idx], dtype=np.complex128).reshape(-1, 1)

    return v_null, A, B, max_eigen_value
