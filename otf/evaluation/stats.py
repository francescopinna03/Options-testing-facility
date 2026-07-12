"""Forecast-comparison statistics for paired model studies.

"Model A beats model B on average" is not a result until the difference
survives its own sampling noise. The out-of-sample studies in this facility
produce PAIRED per-day loss series (same quotes, same shocks), so:

* :func:`dm_test` -- Diebold-Mariano test on the loss differential, HAC
  (Bartlett/Newey-West) variance to survive volatility clustering;
* :func:`block_bootstrap_ci` -- moving-block bootstrap CI for the mean
  differential, preserving serial dependence between neighbouring days;
* :func:`latex_table` -- paste-ready booktabs table for the write-up.

Ported from the Stocks repository (sim/benchmarks.py, apps/thesis_report.py).
"""

from __future__ import annotations
import math
import random
from typing import List, Sequence, Tuple

__all__ = ["dm_test", "block_bootstrap_ci", "latex_table"]


def dm_test(deltas: Sequence[float], lag: int = 2) -> Tuple[float, float]:
    """Diebold-Mariano test on a paired loss-differential series.

    Returns ``(statistic, two_sided_p)`` under the normal approximation
    with a Bartlett/Newey-West HAC variance up to ``lag`` (observations may
    not overlap, but volatility clustering correlates neighbours).
    Positive statistic = the SECOND forecaster (subtracted one) is better.
    """
    xs = [float(d) for d in deltas]
    n = len(xs)
    if n < 2:
        return 0.0, 1.0
    m = sum(xs) / n
    g0 = sum((x - m) ** 2 for x in xs) / n
    hac = g0
    for l in range(1, min(lag, n - 1) + 1):
        g = sum((xs[i] - m) * (xs[i - l] - m) for i in range(l, n)) / n
        hac += 2.0 * (1.0 - l / (lag + 1.0)) * g
    if hac <= 0.0:
        return 0.0, 1.0
    stat = m / math.sqrt(hac / n)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(stat) / math.sqrt(2.0))))
    return stat, p


def block_bootstrap_ci(deltas: Sequence[float], block: int = 5,
                       n_boot: int = 2000, seed: int = 17,
                       level: float = 0.95) -> Tuple[float, float]:
    """Moving-block bootstrap CI for the MEAN of a (possibly serially
    dependent) day-level series -- e.g. paired RMSE differences. Blocks of
    ``block`` consecutive days are resampled with replacement; the CI is
    the percentile interval of the resampled means."""
    xs = [float(d) for d in deltas]
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    b = max(1, min(block, n))
    starts = n - b + 1
    n_blocks = max(1, math.ceil(n / b))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        acc: List[float] = []
        for _ in range(n_blocks):
            s = rng.randrange(starts)
            acc.extend(xs[s:s + b])
        means.append(sum(acc[:n]) / min(len(acc), n))
    means.sort()
    lo = means[int((1.0 - level) / 2.0 * n_boot)]
    hi = means[min(n_boot - 1, int((1.0 + level) / 2.0 * n_boot))]
    return lo, hi


def latex_table(label: str, caption: str, headers: Sequence[str],
                rows: Sequence[Sequence]) -> str:
    """Booktabs LaTeX table, escaped and paste-ready."""
    def esc(s):
        return str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    cols = "l" + "r" * (len(headers) - 1)
    lines = [f"% --- {label} ---",
             r"\begin{table}[htbp]\centering",
             rf"\caption{{{esc(caption)}}}\label{{tab:{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(esc(h) for h in headers) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(esc(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)
