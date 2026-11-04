"""Processing of Teledyne RDI ADCP data read with ``read_instruments.read_pd0``.

- ``depth``: water depth and boat speed from the bottom track;
- ``profiles``: earth velocities and quality control (ship coordinates);
- ``dissipation``: TKE dissipation rate, structure function method;
- ``discharge``: transect discharge, cross-product method;
- ``positions``: positions and speed from a GPS track;
- ``clock``: ADCP clock offset from boat speeds;
- ``bathymetry``: soundings, gridding, validation, GeoTIFF.
"""

from adcp_tools import bathymetry, clock, depth, discharge, dissipation, positions, profiles
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
    "discharge",
    "dissipation",
    "fit_clock_offset",
    "gps_positions",
    "gps_speed",
    "positions",
    "profiles",
    "water_depth",
]
