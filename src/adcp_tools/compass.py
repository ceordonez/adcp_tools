"""Heading correction of the ADCP compass from the GPS track.

The boat's direction over ground is measured twice: by the bottom track,
rotated to earth with the ADCP compass, and by the GPS track. Their angle is
the correction to add to the compass heading: the magnetic declination (when
the heading is magnetic, as with WinRiver's magnetic variation at 0) plus any
compass error. A moving bed would bias it (the bottom track then
underestimates the boat speed in the flow direction), so it is fitted where
the boat moves fast.
"""

import numpy as np
import pandas as pd


def heading_offset(adcp: pd.DataFrame, gps: pd.DataFrame, min_speed: float = 1.0, bin: str = "30s") -> pd.DataFrame:
    """Angle from the bottom-track boat direction to the GPS direction, per `bin`.

    Parameters
    ----------
    adcp : pandas.DataFrame
        Boat velocity from the bottom track, ``East_ms``/``North_ms`` (and
        ``Heading_deg``), per ensemble, true times.
    gps : pandas.DataFrame
        GPS velocity, ``East_ms``/``North_ms``, on `bin` times
        (``positions.gps_velocity``).
    min_speed : float
        Bins where either speed is below this (m/s) are dropped: the direction
        of a slow boat is noisy.

    Returns
    -------
    pandas.DataFrame
        ``Offset_deg`` (add to the heading; -180 to 180), ``Speed_ms`` (GPS),
        ``Heading_deg`` (mean compass heading), one row per bin. The
        correction is ``circular_median(result["Offset_deg"])``.
    """
    adcp = adcp[~adcp.index.duplicated()].sort_index()
    smooth = adcp[["East_ms", "North_ms"]].rolling(bin, center=True).mean()
    heading = np.degrees(np.arctan2(np.sin(np.radians(adcp["Heading_deg"])).rolling(bin, center=True).mean(),
                                    np.cos(np.radians(adcp["Heading_deg"])).rolling(bin, center=True).mean())) % 360
    # GPS velocity between bins i-1 and i is timed at the start of bin i
    at = gps.index
    a = smooth.reindex(at, method="nearest", tolerance=pd.Timedelta(bin) / 2)
    h = heading.reindex(at, method="nearest", tolerance=pd.Timedelta(bin) / 2)
    speed_adcp = np.hypot(a["East_ms"], a["North_ms"])
    speed_gps = np.hypot(gps["East_ms"], gps["North_ms"])
    angle = np.degrees(np.arctan2(gps["North_ms"], gps["East_ms"]) - np.arctan2(a["North_ms"], a["East_ms"]))
    # Math angle (counter-clockwise) to a heading correction (clockwise)
    offset = (-angle + 180) % 360 - 180
    out = pd.DataFrame({"Offset_deg": offset, "Speed_ms": speed_gps, "Heading_deg": h})
    return out[(speed_adcp >= min_speed) & (speed_gps >= min_speed)].dropna()


def circular_median(degrees) -> float:
    """Median of angles (degrees, -180 to 180), around their circular mean."""
    radians = np.radians(np.asarray(degrees, dtype=float))
    mean = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.degrees(mean + np.median((radians - mean + np.pi) % (2 * np.pi) - np.pi)))


def deviation(heading, coefficients) -> np.ndarray:
    """Heading correction (deg) at compass `heading`: A + sum_k B_k sin(k h) + C_k cos(k h).

    `coefficients` = [A, B1, C1, B2, C2, ...] (deg), as from `fit_deviation`.
    """
    h = np.radians(np.asarray(heading, dtype=float))
    out = np.full(h.shape, float(coefficients[0]))
    for k in range(1, (len(coefficients) - 1) // 2 + 1):
        out = out + coefficients[2 * k - 1] * np.sin(k * h) + coefficients[2 * k] * np.cos(k * h)
    return out


def fit_deviation(offsets: pd.DataFrame, harmonics: int = 2, group: pd.Series | None = None, min_group: int = 60,
                  iterations: int = 5) -> tuple[list[float], dict]:
    """Compass deviation curve, and a constant per group (e.g. per day), from `heading_offset` results.

    The curve (`deviation`, `harmonics` harmonics of the heading) is the
    deviation of a compass on a given mount: hard iron gives the first
    harmonic, soft iron the second. The group constants absorb changes
    between groups (the ADCP remounted, other metal nearby); groups with
    fewer than `min_group` values get none. Fitted by least squares,
    alternating curve and constants, without values more than 3 MADs from
    the fit.

    Returns
    -------
    (list of float, dict)
        Coefficients [A, B1, C1, ...] (deg), and {group: constant (deg)}.
    """
    y = offsets["Offset_deg"].to_numpy(dtype=float)
    h = np.radians(offsets["Heading_deg"].to_numpy(dtype=float))
    columns = [np.ones_like(h)]
    for k in range(1, harmonics + 1):
        columns += [np.sin(k * h), np.cos(k * h)]
    X = np.column_stack(columns)
    groups = pd.Series(group.to_numpy() if group is not None else np.zeros(len(y)), index=offsets.index)
    constants = pd.Series(0.0, index=groups.unique())
    keep = np.ones(len(y), dtype=bool)
    for _ in range(iterations):
        shift = groups.map(constants).to_numpy()
        coefficients = np.linalg.lstsq(X[keep], (y - shift)[keep], rcond=None)[0]
        residual = y - X @ coefficients
        keep = np.abs(residual - shift) <= 3 * 1.4826 * np.median(np.abs(residual - shift)) + 1e-9
        if group is not None:
            medians = pd.Series(residual[keep], index=groups[keep].to_numpy()).groupby(level=0).agg(["median", "size"])
            enough = medians["size"] >= min_group
            # The curve's A carries the common offset: centre the constants of the groups that get one
            centre = medians.loc[enough, "median"].median() if enough.any() else 0.0
            constants = (medians["median"] - centre).where(enough, 0.0).reindex(constants.index, fill_value=0.0)
    return [round(float(c), 2) for c in coefficients], ({k: round(float(v), 2) for k, v in constants.items() if v} if group is not None else {})
