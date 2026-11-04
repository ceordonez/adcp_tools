"""ADCP profiles: earth velocities, water depth and quality control.

Written for WinRiver II files in ship coordinates (the 2025 SFM files): the
four velocity "beams" are x (starboard), y (forward), z (up) and error
velocity, already corrected for pitch and roll by the instrument. Earth
velocities are the ship velocities minus the bottom-track velocity, rotated
by the heading; this reproduces WinRiver II's ASCII east/north/up exactly
(checked on SFM Station_7_0_000). Other coordinate systems (e.g. the beam
coordinates of the BlueMap files) are not handled yet.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import read_instruments as ri

from adcp_tools.depth import water_depth


@dataclass
class Profiles:
    """One ADCP file, ready for analysis. Arrays are (cell, ensemble)."""

    time: pd.DatetimeIndex
    """Ensemble times (UTC, clock corrected)."""
    cell_depth: np.ndarray
    """Depth of each cell centre below the surface (m)."""
    cell_size: float
    water_depth: np.ndarray
    """Bottom-track depth below the surface, mean of the beams (m); NaN without bottom."""
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    """Water velocity relative to the ground (m/s)."""
    boat_east: np.ndarray
    boat_north: np.ndarray
    """Boat velocity over ground, from the bottom track (m/s, per ensemble)."""
    ship: np.ndarray
    """Water velocity in ship coordinates, (4, cell, ensemble): x, y, z, error."""
    good: np.ndarray
    """Cells passing the quality control (see `quality_mask`)."""
    beam_angle: float
    ensembles: pd.DataFrame
    """Per-ensemble values of the file (heading, pitch, roll, temperature...)."""

    @property
    def boat_speed(self) -> np.ndarray:
        return np.hypot(self.boat_east, self.boat_north)


def prepare(pd0: ri.PD0, draft: float, qc: dict, bin_1_dist: float | None = None) -> Profiles:
    """Earth velocities, depths and quality mask of one file.

    Parameters
    ----------
    pd0 : read_instruments.PD0
        The file, clock already corrected.
    draft : float
        Transducer depth below the surface (m).
    qc : dict
        ``CORR_MIN``, ``ECHO_ABS_MIN``, ``PGOOD_MIN``, ``BT_SPEED_MAX``.
    bin_1_dist : float or None
        Distance from the transducer to the centre of bin 1 (m); None uses
        the file's value.
    """
    if pd0.coordinates != "ship":
        raise NotImplementedError(f"{pd0.coordinates} coordinates: only ship coordinates are handled")
    heading = np.radians(pd0.ensembles["Heading_deg"].to_numpy())
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    def to_earth(x, y):
        return x * cos_h + y * sin_h, -x * sin_h + y * cos_h

    bt = pd0.bt_vel
    east, north = to_earth(pd0.vel[0] - bt[0], pd0.vel[1] - bt[1])
    # The bottom track is the bottom's velocity seen from the boat
    boat_east, boat_north = to_earth(-bt[0], -bt[1])
    first = pd0.bin_distance[0] if bin_1_dist is None else bin_1_dist
    cell_depth = draft + first + pd0.cell_size * np.arange(pd0.vel.shape[1])
    # RDI's bottom-track range is vertical already (no cos(beam angle) factor):
    # WinRiver II depths are the range plus the draft
    depth = water_depth(pd0, draft).to_numpy()

    profiles = Profiles(
        time=pd.DatetimeIndex(pd0.ensembles.index),
        cell_depth=cell_depth,
        cell_size=pd0.cell_size,
        water_depth=depth,
        east=east,
        north=north,
        up=pd0.vel[2] - bt[2],
        boat_east=boat_east,
        boat_north=boat_north,
        ship=pd0.vel,
        good=np.ones(pd0.vel.shape[1:], dtype=bool),
        beam_angle=pd0.beam_angle or 20.0,
        ensembles=pd0.ensembles,
    )
    profiles.good = quality_mask(pd0, profiles, draft, **{k.lower(): v for k, v in qc.items()})
    return profiles


def quality_mask(
    pd0: ri.PD0,
    profiles: Profiles,
    draft: float,
    corr_min: float,
    echo_abs_min: float,
    pgood_min: float,
    bt_speed_max: float,
) -> np.ndarray:
    """Cells that pass the quality control, (cell, ensemble).

    A cell is rejected when any beam has a correlation below `corr_min` or an
    echo below `echo_abs_min` (dead or flooded beam), when fewer than
    `pgood_min` % of its pings gave a 4-beam solution, when it lies in the
    side-lobe layer near the bottom, or when the boat moved faster than
    `bt_speed_max` (being driven rather than drifting).

    Side-lobe layer (WinRiver II User Guide p. 259): cells deeper than
    range * cos(beam angle) - one cell, measured from the transducer. Without
    a bottom detection the profile is kept whole.
    """
    good = (pd0.corr >= corr_min).all(axis=0) & (pd0.echo >= echo_abs_min).all(axis=0)
    good &= pd0.pgood[3] >= pgood_min
    good &= np.isfinite(pd0.vel[:3]).all(axis=0)
    cutoff = (profiles.water_depth - draft) * np.cos(np.radians(profiles.beam_angle)) - profiles.cell_size
    from_transducer = profiles.cell_depth - draft
    good &= ~(from_transducer[:, None] > cutoff[None, :])  # NaN cutoff (no bottom) keeps the cells
    fast = profiles.boat_speed > bt_speed_max
    good[:, fast] = False
    return good
