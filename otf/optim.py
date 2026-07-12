"""Generic pure-stdlib Nelder-Mead simplex minimiser.

Shared by every inverse problem in the facility (Heston fit, SVI slice fit,
bridge fit, surface fit): all objectives are deterministic (CRN Monte Carlo or
closed form), so a derivative-free simplex is enough and keeps the numerical
core dependency-free.

Ported from the Stocks repository (core/calibration.py).
"""

from __future__ import annotations
from typing import Callable, List, Sequence, Tuple

__all__ = ["nelder_mead"]


def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    step: float = 0.4,
    tol_x: float = 1e-7,
    tol_f: float = 1e-9,
    max_iter: int = 4000,
) -> Tuple[List[float], float, int, bool]:
    """Minimise ``f`` from ``x0`` with the Nelder-Mead downhill simplex.

    Standard coefficients (reflect 1, expand 2, contract 0.5, shrink 0.5).
    Returns ``(x_best, f_best, iterations, converged)``. ``converged`` is True
    when both the simplex spread in x and the spread in f fall below the
    tolerances before ``max_iter`` is reached.
    """
    n = len(x0)
    if n == 0:
        return [], f(x0), 0, True

    # Initial simplex: perturb each coordinate (relative step if non-zero).
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += step * (abs(pt[i]) if pt[i] != 0.0 else 1.0)
        simplex.append(pt)
    fvals = [f(p) for p in simplex]

    alpha, gamma, beta, sigma = 1.0, 2.0, 0.5, 0.5
    converged = False
    it = 0
    while it < max_iter:
        it += 1
        # Order vertices by ascending function value.
        order = sorted(range(n + 1), key=lambda k: fvals[k])
        simplex = [simplex[k] for k in order]
        fvals = [fvals[k] for k in order]

        # Convergence: both x-spread and f-spread small.
        f_spread = abs(fvals[-1] - fvals[0])
        x_spread = max(
            abs(simplex[-1][j] - simplex[0][j]) for j in range(n)
        )
        if f_spread <= tol_f and x_spread <= tol_x:
            converged = True
            break

        # Centroid of all but the worst vertex.
        centroid = [
            sum(simplex[k][j] for k in range(n)) / n for j in range(n)
        ]
        worst = simplex[-1]

        # Reflection.
        refl = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        f_refl = f(refl)
        if fvals[0] <= f_refl < fvals[-2]:
            simplex[-1], fvals[-1] = refl, f_refl
            continue

        # Expansion.
        if f_refl < fvals[0]:
            exp = [centroid[j] + gamma * (refl[j] - centroid[j]) for j in range(n)]
            f_exp = f(exp)
            if f_exp < f_refl:
                simplex[-1], fvals[-1] = exp, f_exp
            else:
                simplex[-1], fvals[-1] = refl, f_refl
            continue

        # Contraction (outside if reflection improved on worst, else inside).
        if f_refl < fvals[-1]:
            contr = [centroid[j] + beta * (refl[j] - centroid[j]) for j in range(n)]
            f_contr = f(contr)
            if f_contr <= f_refl:
                simplex[-1], fvals[-1] = contr, f_contr
                continue
        else:
            contr = [centroid[j] + beta * (worst[j] - centroid[j]) for j in range(n)]
            f_contr = f(contr)
            if f_contr < fvals[-1]:
                simplex[-1], fvals[-1] = contr, f_contr
                continue

        # Shrink toward the best vertex.
        best = simplex[0]
        for k in range(1, n + 1):
            simplex[k] = [best[j] + sigma * (simplex[k][j] - best[j]) for j in range(n)]
            fvals[k] = f(simplex[k])

    best_k = min(range(n + 1), key=lambda k: fvals[k])
    return simplex[best_k], fvals[best_k], it, converged
