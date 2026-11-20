"""Processing of Teledyne RDI ADCP data read with ``aphys_readers.read_pd0``.

- ``depth``: water depth and boat speed from the bottom track;
- ``transform``: beam, instrument and ship to earth coordinates;
- ``profiles``: earth velocities and quality control (ship or beam coordinates);
- ``dissipation``: TKE dissipation rate, structure function method;
- ``discharge``: transect discharge, cross-product method;
- ``positions``: positions and speed from a GPS track;
- ``clock``: ADCP clock offset from boat speeds;
- ``compass``: heading correction from the GPS track;
- ``station``: records at a fixed point (stationary periods, tidal fits);
- ``bathymetry``: soundings, gridding, validation, GeoTIFF.
"""

from adcp_tools import bathymetry, clock, compass, depth, discharge, dissipation, positions, profiles, station, transform
from adcp_tools.clock import fit_clock_offset
from adcp_tools.depth import boat_speed, water_depth
from adcp_tools.transform import beam_to_instrument
from adcp_tools.positions import gps_speed, gps_velocity
from adcp_tools.positions import positions as gps_positions

__all__ = [
    "bathymetry",
    "beam_to_instrument",
    "boat_speed",
    "clock",
    "compass",
    "depth",
    "discharge",
    "dissipation",
    "fit_clock_offset",
    "gps_positions",
    "gps_speed",
    "gps_velocity",
    "positions",
    "profiles",
    "station",
    "transform",
    "water_depth",
]
