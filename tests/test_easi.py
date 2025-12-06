"""Tests for easi_bss. Run with: pytest tests/"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easi_bss import (
    train_easi_sgd,
    train_easi_adam,
    zero_mean_unit_var,
    snr_db,
    best_scale,
    resolve_permutation_and_scaling,
    create_mixtures,
    easi_score_function,
)


class TestUtils:

    def test_zero_mean_unit_var(self):
        """Test normalization function."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        x_norm = zero_mean_unit_var(x)
        
        assert np.abs(x_norm.mean()) < 1e-10, "Mean should be ~0"
        assert np.abs(x_norm.var() - 1) < 1e-10, "Variance should be ~1"

    def test_zero_mean_unit_var_constant(self):
        """Test normalization with constant input."""
        x = np.ones(10)
        x_norm = zero_mean_unit_var(x)
        
        assert np.allclose(x_norm, 0), "Constant input should give zeros"

    def test_snr_db_perfect(self):
        """Test SNR with identical signals."""
        s = np.random.randn(100)
        snr = snr_db(s, s)
        
        assert snr > 100, "Identical signals should have very high SNR"

    def test_snr_db_noisy(self):
        """Test SNR with noisy signal."""
        s_true = np.sin(np.linspace(0, 2*np.pi, 100))
        s_est = s_true + 0.1 * np.random.randn(100)
        snr = snr_db(s_true, s_est)
        
        assert snr > 0, "Low noise should give positive SNR"
        assert snr < 40, "Some noise should limit SNR"

    def test_best_scale(self):
        """Test optimal scaling computation."""
        a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        b = a * 2  # b is a scaled version of a
        
        alpha = best_scale(a, b)
        assert np.abs(alpha - 0.5) < 1e-10, "Scale should be 0.5"


class TestAlgorithm:

    def test_easi_score_function_shape(self):
        """Test score function output shape."""
        y = np.array([0.5, -0.3])
        g = easi_score_function(y)
        
        assert g.shape == (2,), "Score function should return (2,) array"

    def test_create_mixtures_default(self):
        """Test mixture creation with default matrix."""
        S = np.random.randn(2, 100)
        X, A = create_mixtures(S)
        
        assert X.shape == S.shape, "Mixed signals should have same shape"
        assert A.shape == (2, 2), "Mixing matrix should be 2x2"
        assert np.allclose(X, A @ S), "X should equal A @ S"

    def test_create_mixtures_custom(self):
        """Test mixture creation with custom matrix."""
        S = np.random.randn(2, 100)
        A_custom = np.array([[1, 0.5], [0.5, 1]])
        X, A = create_mixtures(S, A_custom)
        
        assert np.allclose(A, A_custom), "Should use provided matrix"

    def test_train_easi_sgd_output_shape(self):
        """Test EASI-SGD output shapes."""
        np.random.seed(42)
        S = np.random.randn(2, 100)
        X, _ = create_mixtures(S)
        
        B, loss_history = train_easi_sgd(X, iterations=100, verbose=False)
        
        assert B.shape == (2, 2), "Demixing matrix should be 2x2"
        assert len(loss_history) > 0, "Should have loss history"
        assert 'total_loss' in loss_history[0], "Loss dict should have total_loss"

    def test_train_easi_adam_output_shape(self):
        """Test EASI-Adam output shapes."""
        np.random.seed(42)
        S = np.random.randn(2, 100)
        X, _ = create_mixtures(S)
        
        B, loss_history = train_easi_adam(X, iterations=100, verbose=False)
        
        assert B.shape == (2, 2), "Demixing matrix should be 2x2"
        assert len(loss_history) > 0, "Should have loss history"


class TestIntegration:

    def test_simple_separation(self):
        """Test that EASI can separate simple mixed signals."""
        np.random.seed(42)
        
        # Create independent sources
        n = 500
        s1 = np.random.laplace(0, 1, n)
        s2 = np.random.uniform(-1, 1, n)
        s1 = zero_mean_unit_var(s1)
        s2 = zero_mean_unit_var(s2)
        S = np.vstack([s1, s2])
        
        # Mix
        X, A = create_mixtures(S)
        
        # Separate
        B, _ = train_easi_adam(X, iterations=5000, verbose=False)
        Y = B @ X
        
        # Evaluate
        snr1, snr2, y1, y2 = resolve_permutation_and_scaling(s1, s2, Y[0], Y[1])
        avg_snr = (snr1 + snr2) / 2
        
        assert avg_snr > 5, f"Average SNR should be > 5 dB, got {avg_snr:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
