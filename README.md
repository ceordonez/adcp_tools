# adcp-tools

Processing of Teledyne RDI ADCP data read with
[`read_instruments`](../read_instruments) (`read_pd0`), shared by the
projects that survey with the ADCP (SFM, BlueMap).

| Module | What |
| --- | --- |
| `depth` | `water_depth` (bottom-track range + draft, optional tilt correction), `boat_speed` (bottom track, any coordinate system) |
| `transform` | `beam_to_instrument` (with 3-beam solutions), `instrument_to_earth` (heading, pitch, roll), `ship_to_earth` (RDI coordinate transformation manual) |
| `profiles` | `subset` (some ensembles of a file), `prepare` (earth velocities, depth, cell depths of one file), `quality_mask` (correlation, echo, percent good, side lobe, boat speed); ship or beam coordinates |
| `dissipation` | `dissipation`: TKE dissipation rate per window, structure function method (Wiles et al. 2006; McGinnis et al. 2014) |
| `discharge` | `transect_discharge` (cross-product method, power-law top/bottom, edges), `downstream_bearing`, `straight_legs` (straight runs of the boat in a long file), `merge_legs` (join legs cut by a stop), `flow_axis` |
| `positions` | `positions` (GPS position at given times, without bridging gaps), `gps_speed`, `gps_velocity` |
| `clock` | `fit_clock_offset`: ADCP clock offset by matching boat speed to GPS speed |
| `compass` | `heading_offset` (GPS minus bottom-track boat direction), `fit_deviation` / `deviation` (compass deviation curve plus constants per day), `circular_median` |
| `bathymetry` | `soundings`, `tide_anomaly`, `to_points`, `track_mask`, `thin`, `interpolate`, `cross_validate`, `crossovers`, `write_geotiff` |

What stays in each project: which files to read, the clock offset and the
draft (config), the GPS and tide sources, the water mask, the figures.

## Install

```
uv add --editable /path/to/adcp_tools            # extras: adcp-tools[tide], adcp-tools[kriging]
```

## Usage

```python
import read_instruments as ri
from adcp_tools import bathymetry, boat_speed, fit_clock_offset, gps_positions, gps_speed, water_depth

pd0 = ri.read_pd0("T10_0_000.PD0", utc_offset="UTC")
gps = ri.read_gps("GPS/", resample=None)

# Clock: true time = ADCP time + best offset
fit = fit_clock_offset(boat_speed(pd0), gps_speed(gps), search=("-2h", "2h"))
offset = fit["r"].idxmax()

depth = water_depth(pd0, draft=0.30, tilt_correction=True)
depth.index = depth.index + offset
found = bathymetry.soundings(depth, gps_positions(depth.index, gps, "30s"), tide=None, bin_seconds=5)

points = bathymetry.to_points(found, "EPSG:32628")
mask = bathymetry.track_mask(points, buffer=250, land=land_polygons, keep=50)  # or a water polygon
grid, transform = bathymetry.interpolate(points, mask, pixel=25, method="linear", max_distance=250)
bathymetry.write_geotiff("bathymetry.tif", grid, transform, "EPSG:32628")
```

## Choices

- **Depth** = mean bottom-track range of the beams + draft, without a
  cos(beam angle) factor: RDI's range is vertical already (WinRiver II gives
  the same).
- **Tide**: depth at mean level = measured - anomaly (`tide_anomaly` predicts
  the anomaly with utide outside the gauge record).
- **Thinning**: soundings are averaged per grid cell before interpolating,
  so stops (anchored stations, drifts) do not outweigh the transects.
- **Validation**: `cross_validate` leaves out every 5th group (a file, or a
  block of minutes); `crossovers` gives the depth differences between surveys
  of the same cell, mostly the tide when it is not corrected.
- **Velocities** (`profiles`): ship coordinates (SFM 2025) reproduce
  WinRiver II exactly. Beam coordinates (BlueMap Sentinel V, 25 degree beams,
  to pass as `beam_angle`) reproduce WinRiver's boat velocity and 90.5% of
  its water cells; the rest differ by WinRiver's bin mapping, not applied
  here, which adds no bias (see the `profiles` docstring).

## Tests

```
uv run pytest
```
