"""Water depth and boat velocity from the ADCP bottom track.

Both work in any coordinate system of the PD0 file (beam, instrument, ship,
earth): the bottom-track range does not depend on it, and the horizontal boat
speed is the same in every frame.
"""

import warnings

import numpy as np
import pandas as pd
from read_instruments import PD0

from adcp_tools.transform import beam_to_instrument


def water_depth(pd0: PD0, draft: float, tilt_correction: bool = False) -> pd.Series:
    """Water depth below the surface at each ensemble (m); NaN without a bottom.

    Depth = mean of the beams' bottom-track ranges + `draft` (transducer depth
    below the surface). RDI's range is vertical already, so there is no
    cos(beam angle) factor: WinRiver II depths are the range plus the draft.
    With `tilt_correction`, the range is multiplied by cos(pitch) cos(roll).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # ensembles without a bottom give NaN
        vertical = np.nanmean(pd0.bt_range, axis=0)
    if tilt_correction:
        e = pd0.ensembles
        vertical = vertical * np.cos(np.radians(e["Pitch_deg"].to_numpy())) * np.cos(np.radians(e["Roll_deg"].to_numpy()))
    return pd.Series(vertical + draft, index=pd0.time, name="Depth_measured_m")


def boat_speed(pd0: PD0, beam_angle: float | None = None) -> pd.Series:
    """Horizontal boat speed over ground from the bottom track (m/s).

    Beam velocities are transformed to the instrument frame first, with
    `beam_angle` (degrees; default the file's, else 20); in the other frames
    the first two components are horizontal (to within the tilt).
    """
    bt = pd0.bt_vel
    if pd0.coordinates == "beam":
        bt = beam_to_instrument(bt, beam_angle or pd0.beam_angle or 20.0)
    return pd.Series(np.hypot(bt[0], bt[1]), index=pd0.time, name="BoatSpeed_ms")
