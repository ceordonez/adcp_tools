"""Processing of Teledyne RDI ADCP data read with ``read_instruments.read_pd0``.

- ``depth``: water depth and boat speed from the bottom track;
- ``positions``: positions and speed from a GPS track;
- ``clock``: ADCP clock offset from boat speeds;
- ``bathymetry``: soundings, gridding, validation, GeoTIFF.
"""

from adcp_tools import bathymetry, clock, depth, positions
from adcp_tools.clock import fit_clock_offset
from adcp_tools.depth import beam_to_instrument, boat_speed, water_depth
from adcp_tools.positions import gps_speed
from adcp_tools.positions import positions as gps_positions

__all__ = [
    "bathymetry",
    "beam_to_instrument",
    "boat_speed",
    "clock",
    "depth",
    "fit_clock_offset",
    "gps_positions",
    "gps_speed",
    "positions",
    "water_depth",
]
