"""River discharge from ADCP transects (moving-boat, cross-product method).

For each ensemble and valid cell, the discharge through the boat track is
(w x b)_z dz dt, with w the water and b the boat velocity over ground; this
needs no assumed transect direction. The unmeasured layers are extrapolated
with a power law fitted to each profile, and the unmeasured edges with
Q = C V d L (as the SFM project's earlier process_adcp.power_method and
shore_flow did).

Sign: positive towards `downstream_bearing`, whichever bank the transect
started from. A meandering channel needs a bearing per station
(`downstream_bearing`): the principal axis of the flow measured there,
oriented towards the mouth.
"""

import warnings

import numpy as np
import pandas as pd

from adcp_tools.profiles import Profiles


def bearing(east, north):
    """Compass bearing (degrees) of a vector."""
    return (np.degrees(np.arctan2(east, north)) + 360) % 360


def downstream_bearing(profiles: list[Profiles], station: tuple[float, float], mouth: tuple[float, float]) -> float:
    """Downstream direction at a station (degrees).

    The channel axis is the principal axis of the depth-averaged velocities of
    all its transects (tidal flow runs along the channel, both ways); of its
    two directions, the one closer to the bearing from the `station` to the
    `mouth` (lat, lon) is downstream.
    """
    with warnings.catch_warnings():
        # Ensembles without valid cells give NaN, dropped below
        warnings.simplefilter("ignore", RuntimeWarning)
        east = np.concatenate([np.nanmean(np.where(p.good, p.east, np.nan), axis=0) for p in profiles])
        north = np.concatenate([np.nanmean(np.where(p.good, p.north, np.nan), axis=0) for p in profiles])
    ok = np.isfinite(east) & np.isfinite(north)
    _, vectors = np.linalg.eigh(np.cov(np.vstack([east[ok], north[ok]])))
    axis = bearing(*vectors[:, -1])
    to_mouth = bearing(
        (mouth[1] - station[1]) * np.cos(np.radians(station[0])), mouth[0] - station[0]
    )
    difference = lambda a: abs((a - to_mouth + 180) % 360 - 180)
    return min((axis, (axis + 180) % 360), key=difference)


def ensemble_durations(time: pd.DatetimeIndex) -> np.ndarray:
    """Duration of each ensemble (s): the time to the next one, the last repeats."""
    dt = np.diff(time.asi8) / 1e9
    return np.append(dt, np.median(dt) if dt.size else 0.0)


def power_law_layers(q: np.ndarray, height: np.ndarray, depth: float, top: float, bottom: float, b: float):
    """Unmeasured top and bottom discharge per unit time and width of one profile.

    Fits q = a h^b (h: height above the bed) to the measured cells by least
    squares, then integrates it from the top of the measured layer to the
    surface and from the bed to its bottom. `top`/`bottom` are the depths
    of the upper and lower edges of the measured layer.
    """
    ok = np.isfinite(q) & (height > 0)
    if ok.sum() < 2:
        return 0.0, 0.0
    hb = height[ok] ** b
    a = (q[ok] * hb).sum() / (hb**2).sum()
    integral = lambda h: a * h ** (b + 1) / (b + 1)
    return integral(depth) - integral(depth - top), integral(depth - bottom)


def transect_discharge(
    profiles: Profiles,
    downstream_bearing: float,
    power_exponent: float = 1 / 6,
    edge_distance: tuple[float, float] = (30.0, 30.0),
    edge_coefficient: float = 0.91,
    edge_ensembles: int = 10,
) -> dict:
    """Discharge through one transect (m3/s).

    Returns a dict: ``Q_m3s`` (total), ``Qm_m3s`` (measured), ``Qt_m3s``
    (top), ``Qb_m3s`` (bottom), ``Qe_m3s`` (edges), ``Length_m`` (track
    length), ``Area_m2`` (measured section), ``Depth_m`` (mean), and
    ``Datetime`` (mid time).
    """
    with warnings.catch_warnings():
        # Transects without valid cells near an edge give empty means
        warnings.simplefilter("ignore", RuntimeWarning)
        return _transect_discharge(profiles, downstream_bearing, power_exponent, edge_distance, edge_coefficient, edge_ensembles)


def _transect_discharge(profiles, downstream_bearing, power_exponent, edge_distance, edge_coefficient, edge_ensembles):
    dt = ensemble_durations(profiles.time)
    w_e = np.where(profiles.good, profiles.east, np.nan)
    w_n = np.where(profiles.good, profiles.north, np.nan)
    cross = w_e * profiles.boat_north[None] - w_n * profiles.boat_east[None]  # (cell, ens), m2/s2
    depth = profiles.water_depth
    dz = profiles.cell_size
    with_bottom = np.isfinite(depth) & np.isfinite(profiles.boat_east)

    q_measured = q_top = q_bottom = 0.0
    for i in np.where(with_bottom)[0]:
        q = cross[:, i]
        valid = np.isfinite(q)
        if not valid.any():
            continue
        q_measured += np.nansum(q) * dz * dt[i]
        cells = profiles.cell_depth[valid]
        top, bottom = cells.min() - dz / 2, cells.max() + dz / 2
        qt, qb = power_law_layers(q, depth[i] - profiles.cell_depth, depth[i], top, bottom, power_exponent)
        q_top += qt * dt[i]
        q_bottom += qb * dt[i]

    # Edges: mean velocity normal to the track times depth times distance to the bank
    track = np.array([np.nanmean(profiles.boat_east), np.nanmean(profiles.boat_north)])
    normal = np.array([track[1], -track[0]]) / np.hypot(*track)  # right of the track
    q_edges = 0.0
    for part, distance in zip((slice(0, edge_ensembles), slice(-edge_ensembles, None)), edge_distance):
        v = np.nanmean(w_e[:, part]) * normal[0] + np.nanmean(w_n[:, part]) * normal[1]
        d = np.nanmean(depth[part])
        if np.isfinite(v) and np.isfinite(d):
            q_edges += edge_coefficient * v * d * distance
    # (w x b)_z is positive for flow to the right of the track: the edges use the same sign

    down = np.radians(downstream_bearing)
    d_hat = np.array([np.sin(down), np.cos(down)])  # east, north
    sign = np.sign(d_hat[0] * track[1] - d_hat[1] * track[0])  # (d x b)_z
    length = np.nansum(profiles.boat_speed * dt)
    q = sign * np.array([q_measured, q_top, q_bottom, q_edges])
    return {
        "Datetime": profiles.time[len(profiles.time) // 2],
        "Q_m3s": q.sum(),
        "Qm_m3s": q[0],
        "Qt_m3s": q[1],
        "Qb_m3s": q[2],
        "Qe_m3s": q[3],
        "Length_m": length,
        "Area_m2": np.nansum(profiles.good[:, with_bottom]) * dz * np.nanmean(profiles.boat_speed * dt),
        "Depth_m": np.nanmean(depth),
    }
