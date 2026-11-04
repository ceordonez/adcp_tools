"""ADCP records at a fixed point: stationary periods and tidal fits.

An ADCP on an anchored (or moored) boat measures, at one place, the water
depth (which rises and falls with the tide) and the current profile:
`stationary_periods` finds such periods in a file, `harmonic_fit` fits the
mean plus tidal constituents to a series.
"""

import numpy as np
import pandas as pd

from adcp_tools.profiles import Profiles

# Periods (h) of the main tidal constituents
PERIODS_H = {"M2": 12.4206012, "S2": 12.0, "K1": 23.9344696, "O1": 25.8193417, "M4": 6.2103006, "MS4": 6.1033392}


def stationary_periods(profiles: Profiles, max_speed: float = 0.3, min_duration: str = "4h", smooth: str = "10min") -> list[slice]:
    """Periods where the boat stayed in place, as slices of the file's ensembles.

    The bottom-track boat speed, as a running median over `smooth`, stays
    below `max_speed` (m/s) for at least `min_duration`. An anchored boat
    swings around its anchor at a few cm/s; the median ignores short moves.
    """
    speed = pd.Series(profiles.boat_speed, index=profiles.time)
    speed = speed[~speed.index.duplicated()].reindex(profiles.time, method="nearest")
    still = (speed.rolling(smooth, center=True, min_periods=3).median() < max_speed).to_numpy()
    periods, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if profiles.time[i - 1] - profiles.time[start] >= pd.Timedelta(min_duration):
                periods.append(slice(start, i))
            start = None
    return periods


def harmonic_fit(series: pd.Series, constituents: list[str]) -> dict:
    """Least-squares fit of mean + constituents to a series (any regular or irregular times).

    Returns ``mean``, and per constituent ``<name>_amp`` and ``<name>_peak``
    (the time of the constituent's first maximum after the series start),
    plus ``rmse`` of the residual and ``n``. Phases are not referred to
    Greenwich (no astronomical argument): the peak times are what a model
    run for the same dates is compared with. Constituents closer in
    frequency than 1/record length cannot be separated (Rayleigh criterion):
    M2 and M4 need 12.42 h, M2 and S2 two weeks.
    """
    data = series.dropna()
    t0 = data.index[0]
    hours = (data.index - t0).total_seconds().to_numpy() / 3600
    columns = [np.ones_like(hours)]
    for name in constituents:
        w = 2 * np.pi / PERIODS_H[name]
        columns += [np.cos(w * hours), np.sin(w * hours)]
    X = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(X, data.to_numpy(dtype=float), rcond=None)
    out = {"mean": float(coef[0])}
    for k, name in enumerate(constituents):
        a, b = coef[1 + 2 * k], coef[2 + 2 * k]
        period = PERIODS_H[name]
        # a cos(wt) + b sin(wt) = A cos(w (t - t_peak))
        t_peak = (np.arctan2(b, a) / (2 * np.pi) * period) % period
        out[f"{name}_amp"] = float(np.hypot(a, b))
        out[f"{name}_peak"] = t0 + pd.Timedelta(hours=float(t_peak))
    out["rmse"] = float(np.sqrt(np.mean((data.to_numpy() - X @ coef) ** 2)))
    out["n"] = len(data)
    return out


def principal_axis(east: np.ndarray, north: np.ndarray) -> float:
    """Bearing (deg, 0-180) of the principal axis of velocity vectors (second moments about zero)."""
    ok = np.isfinite(east) & np.isfinite(north)
    e, n = east[ok], north[ok]
    m = np.array([[np.mean(e * e), np.mean(e * n)], [np.mean(e * n), np.mean(n * n)]])
    _, vectors = np.linalg.eigh(m)
    return float((np.degrees(np.arctan2(*vectors[:, -1])) + 360) % 180)
