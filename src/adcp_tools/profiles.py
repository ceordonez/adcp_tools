"""ADCP profiles: earth velocities, water depth and quality control.

Two coordinate systems of WinRiver II PD0 files are handled:

- **ship** (the SFM 2025 files): the four velocity "beams" are x
  (starboard), y (forward), z (up) and error velocity, already corrected for
  pitch and roll by the instrument. Earth velocities are the ship velocities
  minus the bottom-track velocity, rotated by the heading; this reproduces
  WinRiver II's ASCII east/north/up exactly (checked on SFM Station_7_0_000).
- **beam** (the BlueMap Sentinel V files): beam velocities are screened per
  beam, combined into instrument x, y, z (3-beam solutions where one beam is
  missing) and rotated by heading, pitch and roll (``transform``). The
  Sentinel V files do not state their beam angle (25 degrees): pass it.

  Checked against WinRiver II's Classic ASCII output of BlueMap
  T9_0_001_26-01-29 (5789 ensembles, 58021 cells): the boat velocity agrees
  to 0.03 mm/s (median), and 90.5% of the water cells to the 0.1 cm/s
  rounding of the export. The other cells differ because WinRiver applies
  bin mapping (taking, when the head is tilted, one beam's velocity from the
  neighbouring cell), whose rule for this instrument is not documented and
  is not applied here; it swaps single-ping noise between cells without a
  bias (mean difference -0.07 cm/s; depth-averaged 1-min means within 0.4
  cm/s, for flows of ~30 cm/s). With 20 or 30 degrees instead of 25, the
  boat velocity is off by 11-16 cm/s.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import read_instruments as ri

from adcp_tools import transform
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
    """Water velocity relative to the instrument in its own frame, (4, cell,
    ensemble): x, y, z, error. Ship coordinates as recorded, or instrument
    coordinates computed from the beams (not corrected for tilt)."""
    good: np.ndarray
    """Cells passing the quality control (see `quality_mask`)."""
    beam_angle: float
    ensembles: pd.DataFrame
    """Per-ensemble values of the file (heading, pitch, roll, temperature...)."""

    @property
    def boat_speed(self) -> np.ndarray:
        return np.hypot(self.boat_east, self.boat_north)


def prepare(
    pd0: ri.PD0,
    draft: float,
    qc: dict,
    bin_1_dist: float | None = None,
    beam_angle: float | None = None,
    three_beam: bool = True,
) -> Profiles:
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
    beam_angle : float or None
        Beam angle from the vertical (degrees); None uses the file's value
        (Sentinel V files do not state it: 25).
    three_beam : bool
        Beam coordinates: solve cells missing one beam with the other three.
    """
    angle = beam_angle if beam_angle is not None else pd0.beam_angle
    first = pd0.bin_distance[0] if bin_1_dist is None else bin_1_dist
    cell_depth = draft + first + pd0.cell_size * np.arange(pd0.vel.shape[1])
    # RDI's bottom-track range is vertical already (no cos(beam angle) factor):
    # WinRiver II depths are the range plus the draft
    depth = water_depth(pd0, draft).to_numpy()
    e = pd0.ensembles
    heading = e["Heading_deg"].to_numpy()
    options = {k.lower(): v for k, v in qc.items()}

    if pd0.coordinates == "ship":
        bt = pd0.bt_vel
        east, north = transform.ship_to_earth(pd0.vel[:2] - bt[:2, None, :], heading)
        # The bottom track is the bottom's velocity seen from the boat
        boat_east, boat_north = transform.ship_to_earth(-bt[:2], heading)
        up = pd0.vel[2] - bt[2]
        frame = pd0.vel
        angle = angle or 20.0
    elif pd0.coordinates == "beam":
        if angle is None:
            raise ValueError("The file does not state its beam angle: pass beam_angle (25 for a Sentinel V)")
        screened = np.where(beam_screen(pd0, options["corr_min"], options["echo_abs_min"], options["pgood_min"]), pd0.vel, np.nan)
        frame = transform.beam_to_instrument(screened, angle, three_beam_solutions=three_beam)
        bt = transform.beam_to_instrument(pd0.bt_vel, angle, three_beam_solutions=three_beam)
        pitch, roll = e["Pitch_deg"].to_numpy(), e["Roll_deg"].to_numpy()
        east, north, up = transform.instrument_to_earth(frame[:3] - bt[:3, None, :], heading, pitch, roll, upward=pd0.upward)
        boat_east, boat_north, _ = transform.instrument_to_earth(-bt[:3], heading, pitch, roll, upward=pd0.upward)
    else:
        raise NotImplementedError(f"{pd0.coordinates} coordinates: only ship and beam coordinates are handled")

    profiles = Profiles(
        time=pd.DatetimeIndex(e.index),
        cell_depth=cell_depth,
        cell_size=pd0.cell_size,
        water_depth=depth,
        east=east,
        north=north,
        up=up,
        boat_east=boat_east,
        boat_north=boat_north,
        ship=frame,
        good=np.ones(pd0.vel.shape[1:], dtype=bool),
        beam_angle=angle,
        ensembles=e,
    )
    if pd0.coordinates == "ship":
        profiles.good = quality_mask(pd0, profiles, draft, **options)
    else:
        profiles.good = np.isfinite(east) & np.isfinite(north) & np.isfinite(up)
        profiles.good &= geometry_mask(profiles, draft, options["bt_speed_max"])
    return profiles


def beam_screen(pd0: ri.PD0, corr_min: float, echo_abs_min: float, pgood_min: float) -> np.ndarray:
    """Beam cells kept, (beam, cell, ensemble): correlation, echo and percent good of each beam."""
    return (pd0.corr >= corr_min) & (pd0.echo >= echo_abs_min) & (pd0.pgood >= pgood_min) & np.isfinite(pd0.vel)


def geometry_mask(profiles: Profiles, draft: float, bt_speed_max: float) -> np.ndarray:
    """Cells above the side-lobe layer, in ensembles where the boat was not too fast.

    Side-lobe layer (WinRiver II User Guide p. 259): cells deeper than
    range * cos(beam angle) - one cell, measured from the transducer. Without
    a bottom detection the profile is kept whole.
    """
    cutoff = (profiles.water_depth - draft) * np.cos(np.radians(profiles.beam_angle)) - profiles.cell_size
    from_transducer = profiles.cell_depth - draft
    good = ~(from_transducer[:, None] > cutoff[None, :])  # NaN cutoff (no bottom) keeps the cells
    good[:, profiles.boat_speed > bt_speed_max] = False
    return good


def quality_mask(
    pd0: ri.PD0,
    profiles: Profiles,
    draft: float,
    corr_min: float,
    echo_abs_min: float,
    pgood_min: float,
    bt_speed_max: float,
) -> np.ndarray:
    """Cells that pass the quality control, (cell, ensemble), ship coordinates.

    A cell is rejected when any beam has a correlation below `corr_min` or an
    echo below `echo_abs_min` (dead or flooded beam), when fewer than
    `pgood_min` % of its pings gave a 4-beam solution, when it lies in the
    side-lobe layer near the bottom, or when the boat moved faster than
    `bt_speed_max` (being driven rather than drifting); see `geometry_mask`.
    In beam coordinates the same thresholds screen each beam instead
    (`beam_screen`), before the beams are combined.
    """
    good = (pd0.corr >= corr_min).all(axis=0) & (pd0.echo >= echo_abs_min).all(axis=0)
    good &= pd0.pgood[3] >= pgood_min
    good &= np.isfinite(pd0.vel[:3]).all(axis=0)
    return good & geometry_mask(profiles, draft, bt_speed_max)


def subset(profiles: Profiles, ensembles: slice | np.ndarray) -> Profiles:
    """The same profiles for some ensembles only (a slice, indices or a boolean mask)."""
    return Profiles(
        time=profiles.time[ensembles],
        cell_depth=profiles.cell_depth,
        cell_size=profiles.cell_size,
        water_depth=profiles.water_depth[ensembles],
        east=profiles.east[:, ensembles],
        north=profiles.north[:, ensembles],
        up=profiles.up[:, ensembles],
        boat_east=profiles.boat_east[ensembles],
        boat_north=profiles.boat_north[ensembles],
        ship=profiles.ship[:, :, ensembles],
        good=profiles.good[:, ensembles],
        beam_angle=profiles.beam_angle,
        ensembles=profiles.ensembles.iloc[ensembles] if not isinstance(ensembles, np.ndarray) or ensembles.dtype != bool
        else profiles.ensembles[ensembles],
    )
