import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from adcp_tools import bathymetry, beam_to_instrument, fit_clock_offset, gps_positions, gps_speed


def test_beam_to_instrument_recovers_horizontal_velocity():
    # A velocity (u, v, w) seen by a convex Janus head at 20 degrees
    theta = np.radians(20)
    u, v, w = 1.2, -0.4, 0.05
    s, c = np.sin(theta), np.cos(theta)
    beams = np.array([-u * s - w * c, u * s - w * c, v * s - w * c, -v * s - w * c])
    # RDI sign convention: beam 1 positive x points away from the velocity
    x, y, z, e = beam_to_instrument(-beams, 20.0)
    assert np.hypot(x, y) == pytest.approx(np.hypot(u, v))
    assert e == pytest.approx(0, abs=1e-12)


def test_positions_do_not_bridge_gaps():
    t = pd.date_range("2026-01-01", periods=5, freq="10s", tz="UTC")
    gps = pd.DataFrame({"Lat_deg": [0, 1, 2, np.nan, 4.0], "Lon_deg": [0, 1, 2, np.nan, 4.0]}, index=t)
    gps = gps.drop(t[3])
    times = pd.DatetimeIndex([t[0] + pd.Timedelta("5s"), t[2] + pd.Timedelta("10s")])
    out = gps_positions(times, gps, max_gap="8s")
    assert out["Lat_deg"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(out["Lat_deg"].iloc[1])  # 10 s from both fixes


def test_clock_offset_recovered():
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-01-01", periods=4 * 3600, freq="1s", tz="UTC")
    # Stops and runs at irregular intervals, as on a survey
    level = rng.choice([0.0, 1.0, 2.5], size=len(t) // 300).repeat(300)
    speed = pd.Series(np.convolve(level, np.ones(60) / 60, mode="same") + rng.normal(0, 0.1, len(t)), index=t).clip(lower=0)
    # Track along the equator at that speed, logged by the GPS at true time
    lon = np.degrees(np.cumsum(speed.to_numpy()) / 6_371_000.0)
    gps = pd.DataFrame({"Lat_deg": 0.0, "Lon_deg": lon}, index=t)
    true_offset = pd.Timedelta("3min")
    adcp = speed.copy()
    adcp.index = adcp.index - true_offset  # ADCP clock behind
    result = fit_clock_offset(adcp, gps_speed(gps), search=("-10min", "10min"), step="5s")
    assert abs(result["r"].idxmax() - true_offset) <= pd.Timedelta("5s")
    assert result["r"].max() > 0.9


def test_linear_grid_reproduces_a_plane():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 1000, 500), rng.uniform(0, 1000, 500)
    depth = 2 + 0.01 * x + 0.005 * y
    points = gpd.GeoDataFrame({"Depth_m": depth, "File": np.arange(500) % 10},
                              geometry=gpd.points_from_xy(x, y), crs="EPSG:32628")
    mask = gpd.GeoDataFrame(geometry=[box(100, 100, 900, 900)], crs="EPSG:32628")
    grid, transform = bathymetry.interpolate(points, mask, pixel=10, method="linear")
    rows, cols = np.nonzero(np.isfinite(grid))
    cx = transform.c + (cols + 0.5) * transform.a
    cy = transform.f + (rows + 0.5) * transform.e
    assert np.allclose(grid[rows, cols], 2 + 0.01 * cx + 0.005 * cy, atol=0.05)
    validation = bathymetry.cross_validate(points, mask, ["linear", "idw"], pixel=10)
    assert validation.loc["linear", "RMSE_m"] < 0.1


def test_track_mask_removes_land_and_crossovers():
    x = np.linspace(0, 1000, 101)
    points = gpd.GeoDataFrame({"Depth_m": np.r_[np.full(101, 5.0), np.full(101, 6.0)], "Survey": np.r_[np.zeros(101), np.ones(101)]},
                              geometry=gpd.points_from_xy(np.r_[x, x], np.zeros(202)), crs="EPSG:32628")
    land = gpd.GeoDataFrame(geometry=[box(500, -1000, 2000, 1000)], crs="EPSG:32628")
    mask = bathymetry.track_mask(points, buffer=50, land=land)
    assert mask.total_bounds[2] == pytest.approx(500)
    kept = bathymetry.track_mask(points, buffer=50, land=land, keep=10)
    assert kept.total_bounds[2] == pytest.approx(1010, abs=1)
    cross = bathymetry.crossovers(points, pixel=20)
    assert np.allclose(cross["Range_m"], 1.0)


def test_beam_to_earth_recovers_a_known_flow():
    from adcp_tools import transform

    rng = np.random.default_rng(2)
    n = 50
    heading, pitch, roll = rng.uniform(0, 360, n), rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)
    truth = np.stack([np.full(n, 0.3), np.full(n, -0.5), np.full(n, 0.02)])  # east, north, up
    # Instrument velocity seen by the head: the inverse (transpose) of the rotation
    unit = np.eye(3)[:, :, None] * np.ones(n)
    columns = [transform.instrument_to_earth(np.r_[unit[:, k]], heading, pitch, roll) for k in range(3)]
    rotation = np.stack(columns, axis=1)  # (earth, instrument, ensemble)
    xyz = np.einsum("jin,jn->in", rotation, truth)
    # Beam velocities of that instrument velocity (convex head, 25 deg, positive towards the head)
    s, c = np.sin(np.radians(25)), np.cos(np.radians(25))
    u = np.array([[-s, 0, -c], [s, 0, -c], [0, s, -c], [0, -s, -c]])
    beams = -(u @ xyz)
    back = transform.instrument_to_earth(transform.beam_to_instrument(beams, 25), heading, pitch, roll)
    assert np.allclose(back[:3], truth, atol=1e-12)
    assert np.allclose(back[3], 0, atol=1e-12)
    # One beam missing: the 3-beam solution gives the same velocity, error unknown
    beams[2, 0] = np.nan
    three = transform.instrument_to_earth(transform.beam_to_instrument(beams, 25, three_beam_solutions=True), heading, pitch, roll)
    assert np.allclose(three[:3, 0], truth[:, 0], atol=1e-12)
    assert np.isnan(three[3, 0])
