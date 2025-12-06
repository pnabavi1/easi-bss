import numpy as np
from typing import Tuple, List, Dict, Callable, Optional


def easi_score_function(y: np.ndarray) -> np.ndarray:
    """Score function g(y). tan+y^2 for super-Gaussian, sin+y for sub-Gaussian."""
    y1, y2 = y[0], y[1]
    y1_clip = np.clip(y1, -1.4, 1.4)  # avoid tan blowing up
    y2_clip = np.clip(y2, -3, 3)
    return np.array([np.tan(y1_clip) + y1**2, np.sin(y2_clip) + y2], dtype=np.float64)


def tanh_score_function(y: np.ndarray) -> np.ndarray:
    """Simpler tanh score function (what FastICA uses)."""
    return np.tanh(y)


def compute_easi_loss(B, X, N, num_samples=1000):
    """Returns (total_loss, decorr_loss, det_loss)."""
    idx = np.random.choice(N, min(num_samples, N), replace=False)
    Y = B @ X[:, idx]

    cov = (Y @ Y.T) / Y.shape[1]
    decorr = np.linalg.norm(cov - np.eye(2), 'fro') ** 2
    det_loss = -np.log(np.abs(np.linalg.det(B)) + 1e-10)

    return decorr + 0.1 * det_loss, decorr, det_loss


def train_easi_sgd(
    X: np.ndarray,
    learning_rate: float = 1e-4,
    iterations: int = 50000,
    loss_interval: int = 100,
    score_function: Optional[Callable] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    EASI with vanilla SGD. Update rule:
    B <- B - lr * (yy^T - I + g(y)y^T - yg(y)^T) @ B
    """
    if score_function is None:
        score_function = easi_score_function

    N = X.shape[1]
    B = np.eye(2)
    loss_history = []

    if verbose:
        print(f"EASI-SGD: lr={learning_rate}, iters={iterations:,}")

    for k in range(iterations):
        x = X[:, k % N]
        y = B @ x

        G = score_function(y).reshape(2, 1)
        term = np.outer(y, y) - np.eye(2) + G @ y.reshape(1, 2) - y.reshape(2, 1) @ G.T
        B = B - learning_rate * (term @ B)

        if k % loss_interval == 0:
            total, decorr, det = compute_easi_loss(B, X, N)
            loss_history.append({
                'iteration': k,
                'total_loss': total,
                'decorrelation_loss': decorr,
                'log_det_loss': det
            })
            if verbose and k % 10000 == 0:
                print(f"  iter {k:6,}: loss={total:.6f}")

    if verbose:
        print("  done.")

    return B, loss_history


def train_easi_adam(
    X: np.ndarray,
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    iterations: int = 50000,
    loss_interval: int = 100,
    score_function: Optional[Callable] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    EASI with Adam. Same update as SGD but with adaptive lr.
    Can use ~10x larger lr and converges faster.
    """
    if score_function is None:
        score_function = easi_score_function

    N = X.shape[1]
    B = np.eye(2)
    m = np.zeros_like(B)
    v = np.zeros_like(B)
    loss_history = []

    if verbose:
        print(f"EASI-Adam: lr={learning_rate}, beta1={beta1}, beta2={beta2}")

    for k in range(iterations):
        x = X[:, k % N]
        y = B @ x

        G = score_function(y).reshape(2, 1)
        term = np.outer(y, y) - np.eye(2) + G @ y.reshape(1, 2) - y.reshape(2, 1) @ G.T
        grad = term @ B

        # adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** (k + 1))
        v_hat = v / (1 - beta2 ** (k + 1))
        B = B - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        if k % loss_interval == 0:
            total, decorr, det = compute_easi_loss(B, X, N)
            loss_history.append({
                'iteration': k,
                'total_loss': total,
                'decorrelation_loss': decorr,
                'log_det_loss': det
            })
            if verbose and k % 10000 == 0:
                print(f"  iter {k:6,}: loss={total:.6f}")

    if verbose:
        print("  done.")

    return B, loss_history


def separate_sources(X, method="adam", **kwargs):
    """Wrapper that returns (Y, B, loss_history)."""
    if method.lower() == "sgd":
        B, hist = train_easi_sgd(X, **kwargs)
    elif method.lower() == "adam":
        B, hist = train_easi_adam(X, **kwargs)
    else:
        raise ValueError(f"unknown method: {method}")
    return B @ X, B, hist
