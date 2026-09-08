"""
linear_probe.py — Linear Probing Module for Latent Internal Plastic State Decodability.

Evaluates whether the latent representations of neural operators physically encode the
unobserved internal plastic offset u_p(t) = int_0^t dot{u}_p(tau) dtau = u(t) - F_R(t)/k_0.

Protocol Requirements:
    - Base model weights are completely frozen (no gradient backpropagation).
    - Closed-form Ridge Regression (alpha = 1.0) is fitted strictly on the training set:
          w = (X^T X + alpha * I)^(-1) X^T y
    - Final evaluation is conducted strictly on the held-out test partition (zero leakage).
    - Computes R^2 coefficient of determination, RMSE, and Pearson correlation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch


class LatentPlasticStateProbe:
    """
    Linear Ridge Regression Probe for Latent Feature Representations.

    Args:
        alpha: L2 regularization penalty for Ridge regression (default: 1.0).
        max_samples_train: Subsample limit for training probe to manage memory.
    """

    def __init__(self, alpha: float = 1.0, max_samples_train: int = 2000):
        self.alpha = alpha
        self.max_samples_train = max_samples_train
        self.weight: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.is_fitted = False

    @staticmethod
    def compute_true_plastic_offset(
        u: np.ndarray,
        f_r: np.ndarray,
        k0: float,
    ) -> np.ndarray:
        """
        Compute the exact theoretical internal plastic offset:
            u_p(t) = u(t) - F_R(t) / k_0
        """
        return u - (f_r / max(1e-4, k0))

    def fit(
        self,
        h_train: np.ndarray,
        up_train: np.ndarray,
    ) -> "LatentPlasticStateProbe":
        """
        Fit linear probe on training latent representations via closed-form Ridge solution.

        Args:
            h_train: Latent representations of shape [N_samples, d_latent, Length].
            up_train: True plastic offsets of shape [N_samples, Length].
        """
        n_samples, d_latent, l_seq = h_train.shape

        # Flatten time and sample dimensions: [N * L, d_latent]
        # Transpose h_train: [N, d, L] -> [N, L, d] -> [N*L, d]
        x_flat = np.transpose(h_train, (0, 2, 1)).reshape(-1, d_latent).astype(np.float64)
        y_flat = up_train.reshape(-1).astype(np.float64)

        # Subsample if dataset is large
        if len(x_flat) > self.max_samples_train * l_seq:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x_flat), size=self.max_samples_train * l_seq, replace=False)
            x_flat = x_flat[idx]
            y_flat = y_flat[idx]

        # Center data for intercept
        x_mean = np.mean(x_flat, axis=0)
        y_mean = float(np.mean(y_flat))
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean

        # Closed-form Ridge: w = (X_c^T X_c + alpha * I)^(-1) X_c^T y_c
        xtx = x_centered.T @ x_centered
        reg = self.alpha * np.eye(d_latent, dtype=np.float64)
        xty = x_centered.T @ y_centered

        self.weight = np.linalg.solve(xtx + reg, xty)
        self.bias = float(y_mean - x_mean @ self.weight)
        self.is_fitted = True
        return self

    def predict(self, x_flat: np.ndarray) -> np.ndarray:
        """Predict linear combination: y = X w + b."""
        if not self.is_fitted or self.weight is None:
            raise RuntimeError("Probe must be fitted before predicting.")
        return x_flat.astype(np.float64) @ self.weight + self.bias

    def evaluate(
        self,
        h_test: np.ndarray,
        up_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate linear probe on held-out test latent representations.

        Args:
            h_test: Latent representations of shape [N_test, d_latent, Length].
            up_test: True plastic offsets of shape [N_test, Length].

        Returns:
            Dict containing R^2, RMSE (mm), and Pearson correlation.
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before evaluation.")

        n_samples, d_latent, l_seq = h_test.shape
        x_flat = np.transpose(h_test, (0, 2, 1)).reshape(-1, d_latent)
        y_flat = up_test.reshape(-1).astype(np.float64)

        # Predict plastic offset
        y_pred = self.predict(x_flat)

        # R^2 calculation
        ss_res = float(np.sum((y_flat - y_pred) ** 2))
        ss_tot = float(np.sum((y_flat - np.mean(y_flat)) ** 2))
        r2 = 1.0 - (ss_res / max(1e-8, ss_tot))

        # RMSE in mm (assuming u is in meters, * 1000)
        rmse_mm = float(np.sqrt(np.mean((y_flat - y_pred) ** 2)) * 1000.0)

        # Pearson correlation
        if np.std(y_pred) > 1e-8 and np.std(y_flat) > 1e-8:
            corr = float(np.corrcoef(y_flat, y_pred)[0, 1])
        else:
            corr = 0.0

        return {
            "r2_score": r2,
            "rmse_mm": rmse_mm,
            "pearson_corr": corr,
            "n_test_samples": n_samples,
        }
