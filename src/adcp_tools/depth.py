"""Water depth and boat velocity from the ADCP bottom track.

Both work in any coordinate system of the PD0 file (beam, instrument, ship,
earth): the bottom-track range does not depend on it, and the horizontal boat
speed is the same in every frame.
"""

import warnings

import numpy as np
import pandas as pd
from read_instruments import PD0


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


def beam_to_instrument(beams: np.ndarray, beam_angle: float, convex: bool = True) -> np.ndarray:
    """Janus 4-beam velocities to instrument x, y, z and error velocity.

    `beams` has the four beams on its first axis. RDI ADCP Coordinate
    Transformation (P/N 951-6079-00), eq. 13.
    """
    theta = np.radians(beam_angle)
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    c = 1.0 if convex else -1.0
    d = a / np.sqrt(2)
    b1, b2, b3, b4 = beams
    return np.stack([
        c * a * (b1 - b2),
        c * a * (b4 - b3),
        b * (b1 + b2 + b3 + b4),
        d * (b1 + b2 - b3 - b4),
    ])


def boat_speed(pd0: PD0) -> pd.Series:
    """Horizontal boat speed over ground from the bottom track (m/s).

    Beam velocities are transformed to the instrument frame first; in the
    other frames the first two components are horizontal (to within the tilt).
    """
    bt = pd0.bt_vel
    if pd0.coordinates == "beam":
        bt = beam_to_instrument(bt, pd0.beam_angle or 20.0)
    return pd.Series(np.hypot(bt[0], bt[1]), index=pd0.time, name="BoatSpeed_ms")
