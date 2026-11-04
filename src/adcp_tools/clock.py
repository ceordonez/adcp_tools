"""Check the ADCP clock against the GPS by matching boat speeds.

The bottom-track boat speed and the GPS speed follow the same accelerations
and stops; the clock offset is the shift of the ADCP times that best
correlates them. This is how the SFM 2025 offset (15 days 01:54) was found.
"""

import numpy as np
import pandas as pd


def fit_clock_offset(
    adcp_speed: pd.Series,
    gps_speed: pd.Series,
    search: tuple[str, str] = ("-2h", "2h"),
    step: str = "5s",
    bin: str = "30s",
) -> pd.DataFrame:
    """Correlation between ADCP and GPS boat speed for each clock offset.

    Parameters
    ----------
    adcp_speed : pandas.Series
        Bottom-track boat speed (m/s), ADCP times (see ``depth.boat_speed``).
    gps_speed : pandas.Series
        GPS speed (m/s) on regular `bin` times (see ``positions.gps_speed``).
    search : (str, str)
        Range of offsets tried (true time = ADCP time + offset).
    step : str
        Spacing of the offsets tried, and of the ADCP speeds compared.
    bin : str
        The ADCP speed is averaged over this window, as the GPS speed is.

    Returns
    -------
    pandas.DataFrame
        ``r`` and ``n`` (number of ADCP times compared) indexed by ``Offset``;
        the best offset is ``result["r"].idxmax()``.
    """
    adcp_speed = adcp_speed.dropna().sort_index()
    adcp_speed = adcp_speed[~adcp_speed.index.duplicated()]
    smooth = adcp_speed.rolling(bin, center=True).mean().resample(step).mean().dropna()
    # resample labels each step by its start; time it at its centre
    smooth.index = smooth.index + pd.Timedelta(step) / 2
    gps_speed = gps_speed.dropna().sort_index()
    t_gps = gps_speed.index.as_unit("ns").asi8.astype(float)
    v_gps = gps_speed.to_numpy()
    t_adcp = smooth.index.as_unit("ns").asi8.astype(float)
    v_adcp = smooth.to_numpy()
    max_gap = pd.Timedelta(bin).value
    offsets = pd.timedelta_range(pd.Timedelta(search[0]), pd.Timedelta(search[1]), freq=step)
    rows = []
    for offset in offsets:
        t = t_adcp + offset.value
        v = np.interp(t, t_gps, v_gps, left=np.nan, right=np.nan)
        i = np.clip(np.searchsorted(t_gps, t), 1, len(t_gps) - 1)
        near = np.minimum(np.abs(t_gps[i] - t), np.abs(t - t_gps[i - 1])) <= max_gap
        ok = np.isfinite(v) & near
        n = int(ok.sum())
        r = np.corrcoef(v_adcp[ok], v[ok])[0, 1] if n > 10 else np.nan
        rows.append((offset, r, n))
    return pd.DataFrame(rows, columns=["Offset", "r", "n"]).set_index("Offset")
