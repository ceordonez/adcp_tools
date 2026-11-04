"""Coordinate transformations of RDI ADCP velocities.

Teledyne RDI, "ADCP Coordinate Transformation: Formulas and Calculations"
(P/N 951-6079-00):

- beam to instrument (section 5.1, eq. 13), with 3-beam solutions where one
  beam is missing (section 5.2): the missing beam is the value that makes the
  error velocity zero, and the error velocity of that cell is unknown;
- instrument to earth (section 5.3, eq. 18): rotation by heading, pitch and
  roll. The pitch of the rotation is the "gimballed" pitch
  P = arctan(tan(tilt1) cos(tilt2)) (section 5.3), the roll + 180 deg for an
  upward-looking head;
- ship to earth: rotation by the heading only (ship coordinates are already
  corrected for pitch and roll by the instrument).

Velocity arrays have the four components (beams, or x/y/z/error) on their
first axis; heading, pitch and roll (degrees) broadcast against the last
axis (the ensembles). Bin mapping (EX command bit 0) is not applied (see
``profiles`` for what that changes against WinRiver II).
"""

import numpy as np


def three_beam(beams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill a single missing beam so that the error velocity is zero.

    Returns the completed beams and a mask of the cells that used a 3-beam
    solution. Cells with two or more missing beams stay missing.
    """
    beams = np.array(beams, dtype=float, copy=True)
    missing = ~np.isfinite(beams)
    one = missing.sum(axis=0) == 1
    b1, b2, b3, b4 = beams
    # e is proportional to b1 + b2 - b3 - b4 = 0
    fill = (
        b3 + b4 - b2,
        b3 + b4 - b1,
        b1 + b2 - b4,
        b1 + b2 - b3,
    )
    for k in range(4):
        use = one & missing[k]
        beams[k][use] = fill[k][use]
    return beams, one


def beam_to_instrument(beams: np.ndarray, beam_angle: float, convex: bool = True, three_beam_solutions: bool = False) -> np.ndarray:
    """Janus 4-beam velocities to instrument x, y, z and error velocity.

    With `three_beam_solutions`, cells missing one beam are solved with the
    other three (error velocity NaN there).
    """
    if three_beam_solutions:
        beams, used = three_beam(beams)
    theta = np.radians(beam_angle)
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    c = 1.0 if convex else -1.0
    d = a / np.sqrt(2)
    b1, b2, b3, b4 = beams
    out = np.stack([
        c * a * (b1 - b2),
        c * a * (b4 - b3),
        b * (b1 + b2 + b3 + b4),
        d * (b1 + b2 - b3 - b4),
    ])
    if three_beam_solutions:
        out[3][used] = np.nan
    return out


def gimballed_pitch(pitch, roll):
    """Pitch for the rotation from the two tilt sensors' readings (degrees)."""
    return np.degrees(np.arctan(np.tan(np.radians(pitch)) * np.cos(np.radians(roll))))


def instrument_to_earth(xyz: np.ndarray, heading, pitch, roll, upward: bool = False, gimbal: bool = True) -> np.ndarray:
    """Instrument x, y, z (and error) to east, north, up (and error)."""
    if gimbal:
        pitch = gimballed_pitch(pitch, roll)
    if upward:
        roll = np.asarray(roll) + 180.0
    h, p, r = (np.radians(np.asarray(v, dtype=float)) for v in (heading, pitch, roll))
    ch, sh, cp, sp, cr, sr = np.cos(h), np.sin(h), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    x, y, z = xyz[0], xyz[1], xyz[2]
    east = (ch * cr + sh * sp * sr) * x + sh * cp * y + (ch * sr - sh * sp * cr) * z
    north = (-sh * cr + ch * sp * sr) * x + ch * cp * y + (-sh * sr - ch * sp * cr) * z
    up = -cp * sr * x + sp * y + cp * cr * z
    return np.stack([east, north, up, *xyz[3:]])


def ship_to_earth(xy: np.ndarray, heading) -> np.ndarray:
    """Ship x (starboard), y (forward) to east, north; other components unchanged."""
    h = np.radians(np.asarray(heading, dtype=float))
    x, y = xy[0], xy[1]
    return np.stack([x * np.cos(h) + y * np.sin(h), -x * np.sin(h) + y * np.cos(h), *xy[2:]])
