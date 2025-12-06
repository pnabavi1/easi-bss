"""Utility functions for BSS experiments."""

import numpy as np
from typing import Tuple
from skimage import color, img_as_float
from skimage.transform import resize


def to_grayscale_float(img):
    """Convert to grayscale float [0,1]."""
    img = img_as_float(img)
    if img.ndim == 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)
    return img


def preprocess_images(img1, img2):
    """Grayscale + resize to match."""
    s1 = to_grayscale_float(img1)
    s2 = to_grayscale_float(img2)
    if s1.shape != s2.shape:
        s2 = resize(s2, s1.shape, anti_aliasing=True)
    return s1, s2


def zero_mean_unit_var(x):
    """Normalize to zero mean, unit variance."""
    x = x.astype(np.float64) - x.mean()
    if x.var() > 0:
        x = x / np.sqrt(x.var())
    return x


def add_noise(Y, scale=0.25):
    """Add Gaussian noise, clip to [0,1]."""
    return np.clip(Y + np.random.normal(0, scale, Y.shape), 0, 1)


def snr_db(s_true, s_est):
    """SNR in dB."""
    sig = np.mean(s_true ** 2)
    noise = np.mean((s_true - s_est) ** 2) + 1e-12
    return 10.0 * np.log10(sig / noise)


def kl_divergence(p, q, eps=1e-10):
    """KL(p || q)."""
    p = (p + eps) / np.sum(p + eps)
    q = (q + eps) / np.sum(q + eps)
    return np.sum(p * np.log(p / q))


def best_scale(a, b):
    """Find alpha that minimizes ||a - alpha*b||^2."""
    return float(np.dot(a, b) / (np.dot(b, b) + 1e-12))


def resolve_permutation_and_scaling(s1, s2, y1, y2):
    """
    Handle ICA permutation/scaling ambiguity.
    Returns (snr1, snr2, y1_aligned, y2_aligned).
    """
    # try both orderings, pick the one with higher total SNR
    a11 = best_scale(s1, y1)
    a22 = best_scale(s2, y2)
    direct = snr_db(s1, a11*y1) + snr_db(s2, a22*y2)

    a12 = best_scale(s1, y2)
    a21 = best_scale(s2, y1)
    swapped = snr_db(s1, a12*y2) + snr_db(s2, a21*y1)

    if swapped > direct:
        return snr_db(s1, a12*y2), snr_db(s2, a21*y1), a12*y2, a21*y1
    return snr_db(s1, a11*y1), snr_db(s2, a22*y2), a11*y1, a22*y2


def prepare_sources(img1, img2):
    """Returns (s1, s2, S, H, W) where S is 2xN source matrix."""
    s1_img, s2_img = preprocess_images(img1, img2)
    H, W = s1_img.shape
    s1 = zero_mean_unit_var(s1_img.ravel())
    s2 = zero_mean_unit_var(s2_img.ravel())
    return s1, s2, np.vstack([s1, s2]), H, W


def create_mixtures(S, A=None):
    """X = A @ S. Default A has cond ~7."""
    if A is None:
        A = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float64)
    return A @ S, A


def get_mixing_matrix_info(A):
    """Return cond, det, eigenvalues."""
    return {
        'condition_number': np.linalg.cond(A),
        'determinant': np.linalg.det(A),
        'eigenvalues': np.linalg.eigvals(A),
        'is_invertible': np.abs(np.linalg.det(A)) > 1e-10
    }
