"""TKE dissipation rate from ADCP profiles, structure function method.

Wiles et al. (2006, GRL 33, L21608), as written in the supporting information
of McGinnis et al. (2014): for each averaging window,

    D(r) = < [v'(z) - v'(z + r)]^2 >        (over cell pairs and time)
    D(r)^(3/2) = 3 eps r + 3 N               (C_v = 3^(1/3))

fitted by least squares over separations r; eps = slope / 3, N = intercept / 3.
A fit is discarded when the noise term dominates, N > mean(D^(3/2)) / 3
(McGinnis et al. 2014, SI Eq. 3).

Choices (they differ from the SFM project's earlier process_adcp_epsilon.py):

- r uses the cell size of the file;
- velocity components are chosen (`components`): in ship coordinates the
  4th "beam" is the error velocity, which should not be averaged in;
- pairs can be limited to a depth range (`depth_range`), to estimate the
  dissipation near the surface instead of over the whole profile.
"""

import warnings

import numpy as np
import pandas as pd

from adcp_tools.profiles import Profiles


def structure_function(fluct: np.ndarray, separations: np.ndarray, pair_ok: np.ndarray) -> np.ndarray:
    """D(r) for each separation (in cells), averaged over valid pairs and times.

    `fluct` is (cell, time); `pair_ok[k]` (cell,) marks the upper cells z
    whose pair (z, z + separations[k]) may be used. NaN with fewer than 3
    values.
    """
    n_cells = fluct.shape[0]
    D = np.full(len(separations), np.nan)
    for k, r in enumerate(separations):
        if r >= n_cells:
            continue
        diff = (fluct[: n_cells - r] - fluct[r:]) ** 2
        diff = diff[pair_ok[k][: n_cells - r]]
        values = diff[np.isfinite(diff)]
        if values.size >= 3:
            D[k] = values.mean()
    return D


def fit(r: np.ndarray, D: np.ndarray) -> tuple[float, float, float]:
    """Dissipation, noise and R^2 of the fit D^(3/2) = 3 eps r + 3 N.

    eps is NaN when fewer than 3 separations are valid, when the slope is not
    positive, or when the noise dominates (McGinnis et al. 2014, SI Eq. 3).
    """
    ok = np.isfinite(D) & (D > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = r[ok], D[ok] ** 1.5
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (residual**2).sum() / total if total > 0 else np.nan
    eps, noise = slope / 3, intercept / 3
    if eps <= 0 or noise > y.mean() / 3:
        return np.nan, noise, r2
    return eps, noise, r2


def mean(values: list[float]) -> float:
    """Mean of the finite values, NaN if there are none."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def dissipation(
    profiles: Profiles,
    avg_period_s: float,
    r_min_bins: int,
    r_max_bins: int,
    components: list[int],
    depth_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Dissipation rate per averaging window.

    Parameters
    ----------
    profiles : adcp_tools.profiles.Profiles
    avg_period_s : float
        Window length (s); the mean of each window is removed.
    r_min_bins, r_max_bins : int
        Separations used, in cells.
    components : list of int
        Ship-coordinate components (0 = x, 1 = y, 2 = z). With vertical
        separations, z gives the longitudinal structure function.
    depth_range : (float, float) or None
        Only pairs whose mid-point is within this depth range (m below the
        surface); None uses the whole profile.

    Returns
    -------
    pandas.DataFrame
        ``Epsilon_Wkg`` (log-mean over components), ``Noise_m3s3``, ``R2``
        (means) and ``n_fits``, indexed by the window centre time.
    """
    dt = np.median(np.diff(profiles.time.asi8)) / 1e9
    n_avg = max(1, int(round(avg_period_s / dt)))
    n_windows = len(profiles.time) // n_avg
    separations = np.arange(r_min_bins, r_max_bins + 1)
    r = separations * profiles.cell_size  # vertical cells: separation is vertical

    depth = profiles.cell_depth
    if depth_range is None:
        pair_ok = np.ones((len(separations), len(depth)), dtype=bool)
    else:
        middle = depth[None, :] + separations[:, None] * profiles.cell_size / 2
        pair_ok = (middle >= depth_range[0]) & (middle <= depth_range[1])

    velocity = np.where(profiles.good[None], profiles.ship[: max(components) + 1], np.nan)
    rows = []
    for w in range(n_windows):
        window = slice(w * n_avg, (w + 1) * n_avg)
        fits = []
        for c in components:
            v = velocity[c, :, window]
            if np.isfinite(v).sum() < (r_max_bins + 1) * 3:
                continue
            with warnings.catch_warnings():
                # Cells without data in the window give NaN means
                warnings.simplefilter("ignore", RuntimeWarning)
                fluct = v - np.nanmean(v, axis=1, keepdims=True)
            fits.append(fit(r, structure_function(fluct, separations, pair_ok)))
        eps = np.array([f[0] for f in fits]) if fits else np.array([np.nan])
        valid = np.isfinite(eps)
        rows.append(
            {
                "Datetime": profiles.time[w * n_avg + n_avg // 2],
                "Epsilon_Wkg": 10 ** np.mean(np.log10(eps[valid])) if valid.any() else np.nan,
                "Noise_m3s3": mean([f[1] for f in fits]),
                "R2": mean([f[2] for f in fits]),
                "n_fits": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("Datetime") if rows else pd.DataFrame(
        columns=["Epsilon_Wkg", "Noise_m3s3", "R2", "n_fits"], index=pd.DatetimeIndex([], name="Datetime", tz="UTC")
    )
