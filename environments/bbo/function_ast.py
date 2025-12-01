"""Random functions from principled mathematical foundations.

Implements f(x) = γ·f_convex(x) + (1-γ)·f_spectral(x) + ε‖x‖²
where components are sampled from well-understood distributions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class FunctionProperties:
    """Analytical properties derived from generating parameters."""
    dimension: int
    smoothness: float          # Matérn ν (higher = smoother)
    convexity_weight: float    # γ ∈ [0,1]
    condition_number: float    # κ = λ_max/λ_min
    length_scale: float
    lipschitz_estimate: float
    n_spectral: int
    n_planes: int

    @property
    def is_convex(self) -> bool:
        return self.convexity_weight > 0.99

    @property
    def is_smooth(self) -> bool:
        return self.smoothness > 10.0

    def theoretical_regret_bound(self, n_queries: int) -> float:
        """Information-theoretic lower bound on achievable regret."""
        T = max(1, n_queries)
        if self.convexity_weight > 0.99:
            return self.condition_number * np.exp(-T / np.sqrt(self.condition_number))
        return self.lipschitz_estimate * np.sqrt(self.dimension / T)


class RandomFunction:
    """A random function built from quadratic + piecewise-linear + spectral components."""

    def __init__(
        self,
        H: np.ndarray,           # Hessian (d×d PSD)
        g: np.ndarray,           # Linear term (d,)
        c: float,                # Constant
        A: np.ndarray,           # Plane normals (n_planes×d)
        b: np.ndarray,           # Plane offsets (n_planes,)
        tau: float,              # Softmax temperature
        omega: np.ndarray,       # Frequencies (n_freq×d)
        alpha: np.ndarray,       # Amplitudes (n_freq,)
        phi: np.ndarray,         # Phases (n_freq,)
        gamma: float,            # Convexity weight
        eps: float,              # Regularization
        props: FunctionProperties,
    ):
        self.H, self.g, self.c = H, g, c
        self.A, self.b, self.tau = A, b, tau
        self.omega, self.alpha, self.phi = omega, alpha, phi
        self.gamma, self.eps = gamma, eps
        self.props = props
        self._d = len(g)

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def dimension(self) -> int:
        return self._d

    def properties(self) -> FunctionProperties:
        return self.props

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        f_quad = 0.5 * x @ self.H @ x + self.g @ x + self.c
        f_pwl = self._logsumexp(self.A @ x + self.b, self.tau) if len(self.b) else 0.0
        f_spec = np.sum(self.alpha * np.cos(self.omega @ x + self.phi)) if len(self.alpha) else 0.0
        return float(self.gamma * (f_quad + f_pwl) + (1 - self.gamma) * f_spec + self.eps * (x @ x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        grad = self.gamma * (self.H @ x + self.g) + 2 * self.eps * x

        if len(self.b):
            w = self._softmax(self.A @ x + self.b, self.tau)
            grad += self.gamma * (self.A.T @ w)

        if len(self.alpha):
            grad += (1 - self.gamma) * (self.omega.T @ (-self.alpha * np.sin(self.omega @ x + self.phi)))

        return grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        hess = self.gamma * self.H + 2 * self.eps * np.eye(self._d)

        if len(self.b):
            w = self._softmax(self.A @ x + self.b, self.tau)
            mu = self.A.T @ w
            hess += self.gamma / self.tau * (self.A.T @ np.diag(w) @ self.A - np.outer(mu, mu))

        if len(self.alpha):
            coeffs = -self.alpha * np.cos(self.omega @ x + self.phi)
            hess += (1 - self.gamma) * (self.omega.T @ np.diag(coeffs) @ self.omega)

        return hess

    @staticmethod
    def _logsumexp(z: np.ndarray, tau: float) -> float:
        z_scaled = z / tau
        return tau * (np.max(z_scaled) + np.log(np.sum(np.exp(z_scaled - np.max(z_scaled)))))

    @staticmethod
    def _softmax(z: np.ndarray, tau: float) -> np.ndarray:
        z_scaled = z / tau
        e = np.exp(z_scaled - np.max(z_scaled))
        return e / np.sum(e)


# === Matrix Utilities ===

def random_psd(d: int, condition: float, rng: np.random.Generator) -> np.ndarray:
    """Sample PSD matrix with given condition number via eigendecomposition."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigvals = np.exp(rng.uniform(0, np.log(condition), d))
    eigvals = (eigvals - eigvals.min() + 1) / eigvals.max() * condition
    return Q @ np.diag(eigvals) @ Q.T


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from O(d)."""
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q @ np.diag(np.sign(np.diag(R)))


def matern_frequencies(d: int, n: int, nu: float, length_scale: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Sample frequencies from Matérn spectral density (Student-t)."""
    z = rng.standard_normal((n, d))
    scale = np.sqrt(2 * nu / rng.chisquare(2 * nu, n))[:, None]
    return z * scale / length_scale
