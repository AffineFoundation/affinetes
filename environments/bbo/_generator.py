"""Seed-based problem generation via maximum-entropy priors.

seed → hyperparameters (max-entropy) → function parameters → f(x) → x*
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
from scipy.optimize import minimize, differential_evolution

from ._function import RandomFunction, FunctionProperties, random_psd, matern_frequencies


@dataclass(frozen=True)
class Hyperparameters:
    """All parameters sampled from maximum-entropy priors."""
    dimension: int
    smoothness: float       # Matérn ν
    length_scale: float     # Correlation length
    convexity: float        # γ ∈ [0,1]
    log_condition: float    # log(κ)
    n_spectral: int
    n_planes: int

    @property
    def condition(self) -> float:
        return np.exp(self.log_condition)

    @property
    def query_budget(self) -> int:
        return max(10, int(10 * self.dimension * np.log1p(self.condition)))


@dataclass
class Problem:
    """A complete optimization problem."""
    function: RandomFunction
    hp: Hyperparameters
    bounds: np.ndarray
    x_opt: np.ndarray
    f_opt: float
    x_init: np.ndarray
    f_init: float
    seed: int

    def spec(self) -> Dict[str, Any]:
        """Problem specification (without revealing optimum)."""
        p = self.function.properties()
        return dict(
            seed=self.seed,
            dimension=p.dimension,
            bounds=self.bounds.tolist(),
            smoothness=p.smoothness,
            convexity_weight=p.convexity_weight,
            condition_number=p.condition_number,
            length_scale=p.length_scale,
            n_spectral=p.n_spectral,
            n_planes=p.n_planes,
            is_convex=p.is_convex,
            is_smooth=p.is_smooth,
            initial_point=self.x_init.tolist(),
            initial_value=self.f_init,
            query_budget=self.hp.query_budget,
        )


# === Maximum-Entropy Priors ===
# These are the ONLY design choices, justified by maximum entropy principle

PRIORS = dict(
    dim_p=0.1,              # Geometric: E[d] ≈ 10
    smooth_shape=2.0,       # Gamma shape
    smooth_scale=2.0,       # Gamma scale
    length_mu=0.0,          # LogNormal: median = 1
    length_sigma=1.0,
    log_kappa_max=10.0,     # LogUniform[0, 10] → κ ∈ [1, 22000]
    spectral_mean=50.0,     # Poisson
    planes_mean=10.0,       # Poisson
)


def _spawn_rngs(seed: int, n: int = 3) -> Tuple[np.random.Generator, ...]:
    """Deterministically split seed into independent RNG streams."""
    seqs = np.random.SeedSequence(seed).spawn(n)
    return tuple(np.random.default_rng(s) for s in seqs)


def sample_hyperparameters(rng: np.random.Generator) -> Hyperparameters:
    """Sample from max-entropy priors."""
    P = PRIORS
    return Hyperparameters(
        dimension=min(100, 1 + rng.geometric(P['dim_p'])),
        smoothness=max(0.5, rng.gamma(P['smooth_shape'], P['smooth_scale'])),
        length_scale=rng.lognormal(P['length_mu'], P['length_sigma']),
        convexity=rng.uniform(0, 1),
        log_condition=rng.uniform(0, P['log_kappa_max']),
        n_spectral=max(1, rng.poisson(P['spectral_mean'])),
        n_planes=rng.poisson(P['planes_mean']),
    )


def sample_function(hp: Hyperparameters, rng: np.random.Generator) -> RandomFunction:
    """Generate random function from hyperparameters."""
    d = hp.dimension

    # Quadratic: ½xᵀHx + gᵀx + c
    H = random_psd(d, hp.condition, rng)
    g = rng.standard_normal(d) * np.sqrt(d)
    c = rng.uniform(-10, 10)

    # Piecewise-linear: softmax of hyperplanes
    if hp.n_planes > 0:
        A = rng.standard_normal((hp.n_planes, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-10
        b = rng.standard_normal(hp.n_planes)
    else:
        A, b = np.zeros((0, d)), np.zeros(0)

    # Spectral: random Fourier features
    if hp.n_spectral > 0:
        omega = matern_frequencies(d, hp.n_spectral, hp.smoothness, hp.length_scale, rng)
        alpha = rng.standard_normal(hp.n_spectral) / np.sqrt(hp.n_spectral)
        phi = rng.uniform(0, 2 * np.pi, hp.n_spectral)
    else:
        omega, alpha, phi = np.zeros((0, d)), np.zeros(0), np.zeros(0)

    # Estimate Lipschitz constant
    L_H = np.linalg.norm(H, 2)
    L_A = np.max(np.linalg.norm(A, axis=1)) if len(A) else 0
    L_spec = np.sum(np.abs(alpha) * np.sum(omega**2, axis=1)) if len(alpha) else 0
    L = hp.convexity * (L_H + L_A) + (1 - hp.convexity) * L_spec

    tau = 1 / np.sqrt(d)
    eps = 0.01 / d

    props = FunctionProperties(
        dimension=d,
        smoothness=hp.smoothness,
        convexity_weight=hp.convexity,
        condition_number=hp.condition,
        length_scale=hp.length_scale,
        lipschitz_estimate=L + 2 * eps,
        n_spectral=hp.n_spectral,
        n_planes=hp.n_planes,
    )

    return RandomFunction(H, g, c, A, b, tau, omega, alpha, phi, hp.convexity, eps, props)


def find_optimum(f: RandomFunction, bounds: np.ndarray, rng: np.random.Generator,
                 n_starts: int = 10) -> Tuple[np.ndarray, float]:
    """Find a high-quality minimum via light-weight multi-start search."""
    d = f.dimension()
    scipy_bounds = list(zip(bounds[:, 0], bounds[:, 1]))

    def _local_minimum(x0: np.ndarray) -> Tuple[np.ndarray, float]:
        res = minimize(
            f.evaluate,
            x0,
            jac=f.gradient,
            method='L-BFGS-B',
            bounds=scipy_bounds,
            options={'maxiter': 200, 'gtol': 1e-8},
        )
        f_val = float(res.fun)
        if not np.isfinite(f_val):
            return x0, float(f.evaluate(x0))
        return res.x, f_val

    starts = [np.zeros(d)] + [rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(max(1, n_starts - 1))]
    best_x, best_f = _local_minimum(starts[0])

    for x0 in starts[1:]:
        x_cand, f_cand = _local_minimum(x0)
        if f_cand < best_f:
            best_x, best_f = x_cand, f_cand

    if f.properties().convexity_weight < 0.9:
        res = differential_evolution(
            f.evaluate,
            scipy_bounds,
            maxiter=max(25, 300 // max(1, d)),
            seed=int(rng.integers(2**31)),
            tol=1e-8,
            polish=True,
        )
        if res.success and np.isfinite(res.fun) and res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)

    return best_x, float(best_f)


def generate(seed: int) -> Problem:
    """Generate a complete optimization problem from seed."""
    hp_rng, func_rng, opt_rng = _spawn_rngs(seed, 3)

    hp = sample_hyperparameters(hp_rng)
    bounds = np.array([[-5 * hp.length_scale, 5 * hp.length_scale]] * hp.dimension)
    func = sample_function(hp, func_rng)
    x_opt, f_opt = find_optimum(func, bounds, opt_rng)

    x_init = np.zeros(hp.dimension)
    f_init = func(x_init)

    return Problem(func, hp, bounds, x_opt, f_opt, x_init, f_init, seed)
