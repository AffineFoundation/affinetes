"""Random functions from principled mathematical foundations.

Implements f(x) = γ·f_convex(x) + (1-γ)·f_spectral(x) + ε‖x‖²
where components are sampled from well-understood distributions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class RandomFunction:
    """Random function = γ·(quadratic + softmax planes) + (1-γ)·spectral + ε‖x‖²."""

    H: np.ndarray           # (d×d) PSD
    g: np.ndarray           # (d,) linear term
    c: float                # constant offset
    A: np.ndarray           # (n_planes×d) plane normals
    b: np.ndarray           # (n_planes,) plane offsets
    tau: float              # softmax temperature
    omega: np.ndarray       # (n_freq×d) frequencies
    alpha: np.ndarray       # (n_freq,) amplitudes
    phi: np.ndarray         # (n_freq,) phases
    gamma: float            # convexity weight
    eps: float              # small strongly-convex regularizer
    props: FunctionProperties

    _d: int = field(init=False, repr=False)
    _has_planes: bool = field(init=False, repr=False)
    _has_spectral: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_d", self.g.shape[0])
        object.__setattr__(self, "_has_planes", self.b.size > 0)
        object.__setattr__(self, "_has_spectral", self.alpha.size > 0)

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def dimension(self) -> int:
        return self._d

    def properties(self) -> FunctionProperties:
        return self.props

    def evaluate(self, x: np.ndarray) -> float:
        x = self._ensure_vector(x)
        f_quad = 0.5 * x @ self.H @ x + self.g @ x + self.c
        f_pwl = self._logsumexp(self.A @ x + self.b, self.tau) if self._has_planes else 0.0
        if self._has_spectral:
            phases = self.omega @ x + self.phi
            f_spec = np.sum(self.alpha * np.cos(phases))
        else:
            f_spec = 0.0
        return float(self.gamma * (f_quad + f_pwl) + (1 - self.gamma) * f_spec + self.eps * (x @ x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_vector(x)
        grad = self.gamma * (self.H @ x + self.g) + 2 * self.eps * x

        if self._has_planes:
            w = self._softmax(self.A @ x + self.b, self.tau)
            grad += self.gamma * (self.A.T @ w)

        if self._has_spectral:
            phases = self.omega @ x + self.phi
            grad += (1 - self.gamma) * (self.omega.T @ (-self.alpha * np.sin(phases)))

        return grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_vector(x)
        hess = self.gamma * self.H + 2 * self.eps * np.eye(self._d)

        if self._has_planes:
            w = self._softmax(self.A @ x + self.b, self.tau)
            mu = self.A.T @ w
            weighted_A = self.A.T * w
            pwl_hess = weighted_A @ self.A - np.outer(mu, mu)
            hess += self.gamma / self.tau * pwl_hess

        if self._has_spectral:
            phases = self.omega @ x + self.phi
            coeffs = -self.alpha * np.cos(phases)
            hess += (1 - self.gamma) * ((self.omega.T * coeffs) @ self.omega)

        return hess

    def _ensure_vector(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        original_shape = x_arr.shape
        x_vec = x_arr.reshape(-1)
        if x_vec.shape[0] != self._d:
            raise ValueError(f"Expected vector of length {self._d}, got shape {original_shape}")
        return x_vec

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
