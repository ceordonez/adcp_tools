"""Positions and speeds from a GPS track (``Lat_deg``, ``Lon_deg``, UTC index)."""

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def positions(times: pd.DatetimeIndex, gps: pd.DataFrame, max_gap: str = "60s") -> pd.DataFrame:
    """GPS position at each time, interpolated linearly between fixes.

    Times further than `max_gap` from the nearest fix get NaN, so gaps in the
    track are not bridged.
    """
    gps = gps[["Lat_deg", "Lon_deg"]].dropna().sort_index()
    gps = gps[~gps.index.duplicated()]
    index = pd.DatetimeIndex(times, name="Datetime")
    if gps.empty:
        return pd.DataFrame({"Lat_deg": np.nan, "Lon_deg": np.nan}, index=index)
    t_gps = gps.index.as_unit("ns").asi8.astype(float)
    t = index.as_unit("ns").asi8.astype(float)
    out = pd.DataFrame(
        {c: np.interp(t, t_gps, gps[c].to_numpy(), left=np.nan, right=np.nan) for c in gps},
        index=index,
    )
    # Distance to the nearest fix
    after = np.clip(np.searchsorted(t_gps, t), 0, len(t_gps) - 1)
    before = np.clip(after - 1, 0, len(t_gps) - 1)
    nearest = np.minimum(np.abs(t_gps[after] - t), np.abs(t - t_gps[before]))
    out.loc[nearest > pd.Timedelta(max_gap).value] = np.nan
    return out


def distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance (m)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


def gps_speed(gps: pd.DataFrame, bin: str = "30s") -> pd.Series:
    """Speed over ground (m/s) between successive `bin` means of the track.

    Bins without a fix are NaN, and so is the speed across them.
    """
    means = gps[["Lat_deg", "Lon_deg"]].resample(bin).mean()
    dt = pd.Timedelta(bin).total_seconds()
    d = distance_m(means["Lat_deg"].shift(), means["Lon_deg"].shift(), means["Lat_deg"], means["Lon_deg"])
    # Speed between bin i-1 and i, timed at the boundary between them
    speed = pd.Series(d / dt, index=means.index, name="GPSSpeed_ms")
    return speed


def gps_velocity(gps: pd.DataFrame, bin: str = "30s") -> pd.DataFrame:
    """East and north velocity over ground (m/s) between successive `bin` means of the track.

    Timed like `gps_speed`: at the boundary between the two bins.
    """
    means = gps[["Lat_deg", "Lon_deg"]].resample(bin).mean()
    dt = pd.Timedelta(bin).total_seconds()
    lat = np.radians(means["Lat_deg"])
    east = np.radians(means["Lon_deg"].diff()) * np.cos(lat) * EARTH_RADIUS_M / dt
    north = np.radians(means["Lat_deg"].diff()) * EARTH_RADIUS_M / dt
    return pd.DataFrame({"East_ms": east, "North_ms": north}, index=means.index)
