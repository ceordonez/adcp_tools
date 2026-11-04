"""Bathymetry from ADCP bottom-track soundings.

Steps:

1. `soundings`: the water depth of each ensemble (``depth.water_depth``),
   optionally reduced to the mean water level with a tide anomaly, positioned
   with the GPS and averaged over short time bins.
2. `to_points`: soundings as points in a projected CRS (metres).
3. `track_mask` (when there is no water polygon of the surveyed area): a
   corridor around the soundings, minus land.
4. `interpolate`: soundings averaged per grid cell (`thin`), then gridded
   inside the mask (linear, cubic, nearest, IDW or kriging).
5. `cross_validate`: error of each method on blocks of soundings left out;
   `crossovers`: depth differences where several surveys overlap (without a
   tide correction, mostly the tide).
6. `write_geotiff`.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


def tide_anomaly(times: pd.DatetimeIndex, level: pd.DataFrame, latitude: float, max_extrapolation: str = "5D") -> pd.DataFrame:
    """Water-level anomaly (m) at `times`, and whether it was predicted.

    `level` has an ``Anomaly_m`` column (water level minus its mean). Within
    its record the anomaly is interpolated linearly. Outside it, up to
    `max_extrapolation` away, it is predicted from a harmonic fit to the
    record (utide, constituents chosen automatically); further away it is NaN.
    """
    import utide

    t_level = level.index.as_unit("ns").asi8.astype(float)
    t = pd.DatetimeIndex(times).as_unit("ns").asi8.astype(float)
    observed = np.interp(t, t_level, level["Anomaly_m"].to_numpy(), left=np.nan, right=np.nan)
    out = pd.DataFrame({"Anomaly_m": observed, "Tide_predicted": False}, index=pd.DatetimeIndex(times))
    outside = np.isnan(observed)
    near = (out.index >= level.index[0] - pd.Timedelta(max_extrapolation)) & (
        out.index <= level.index[-1] + pd.Timedelta(max_extrapolation)
    )
    todo = outside & near
    if todo.any():
        # utide takes naive datetimes (its float input is a different day count)
        coef = utide.solve(
            level.index.tz_localize(None).to_numpy(), level["Anomaly_m"].to_numpy(), lat=latitude,
            method="ols", conf_int="none", trend=False, verbose=False,
        )
        predicted = utide.reconstruct(out.index[todo].tz_localize(None).to_numpy(), coef, verbose=False).h
        out.loc[todo, "Anomaly_m"] = predicted
        out.loc[todo, "Tide_predicted"] = True
    return out


def soundings(
    depth: pd.Series,
    positions: pd.DataFrame,
    tide: pd.DataFrame | None = None,
    bin_seconds: float = 5,
) -> pd.DataFrame:
    """Soundings averaged over `bin_seconds`.

    Parameters
    ----------
    depth : pandas.Series
        Measured water depth (m) per ensemble (``depth.water_depth``).
    positions : pandas.DataFrame
        ``Lat_deg``, ``Lon_deg`` at the same times (``positions.positions``).
    tide : pandas.DataFrame or None
        ``Anomaly_m`` (and optionally ``Tide_predicted``) at the same times;
        the depth at mean water level is then measured - anomaly.

    Returns
    -------
    pandas.DataFrame
        ``Depth_m`` (at mean water level with `tide`, else as measured),
        ``Depth_measured_m``, ``Lat_deg``, ``Lon_deg`` (and the tide columns),
        indexed by time; bins without a bottom or a position are dropped.
    """
    data = pd.DataFrame({"Depth_measured_m": depth.to_numpy()}, index=depth.index)
    data = data.join(positions[["Lat_deg", "Lon_deg"]].set_axis(depth.index))
    if tide is not None:
        data = data.join(tide.set_axis(depth.index))
        data["Depth_m"] = data["Depth_measured_m"] - data["Anomaly_m"]
    else:
        data["Depth_m"] = data["Depth_measured_m"]
    binned = data.resample(f"{bin_seconds}s").mean()
    if "Tide_predicted" in binned:
        binned["Tide_predicted"] = binned["Tide_predicted"] > 0
    return binned.dropna(subset=["Depth_m", "Lat_deg", "Lon_deg"])


def to_points(data: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    """Soundings as points in a projected CRS."""
    points = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["Lon_deg"], data["Lat_deg"]), crs="EPSG:4326")
    return points.to_crs(crs)


def track_mask(points: gpd.GeoDataFrame, buffer: float, land: gpd.GeoDataFrame | None = None, min_area: float = 0.0,
               keep: float = 0.0) -> gpd.GeoDataFrame:
    """Area within `buffer` metres of the soundings, minus `land`.

    For surveys of lines rather than of a whole water body: the grid is then
    only defined near the soundings. Land polygons are rarely accurate in
    narrow channels, so the area within `keep` metres of a sounding stays in
    the mask even where `land` covers it. Parts smaller than `min_area` (m²)
    are dropped. Returns one row per connected part, in the CRS of `points`.
    """
    area = points.geometry.buffer(buffer, quad_segs=4).union_all()
    if land is not None and not land.empty:
        area = area.difference(land.to_crs(points.crs).geometry.union_all())
        if keep > 0:
            area = area.union(points.geometry.buffer(keep, quad_segs=2).union_all())
    parts = gpd.GeoDataFrame(geometry=[area], crs=points.crs).explode(index_parts=False)
    parts = parts[parts.area >= min_area].reset_index(drop=True)
    return parts


def thin(points: gpd.GeoDataFrame, pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """Soundings averaged per `pixel` x `pixel` cell: positions (n, 2) and depths (n).

    Stops (anchored stations, chamber drifts) leave many soundings on the
    same spot; averaging per cell gives each place one value, so they do not
    outweigh the transects, and removes duplicate points for the
    triangulation.
    """
    x, y = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
    key = pd.DataFrame({"i": np.floor(x / pixel), "j": np.floor(y / pixel), "x": x, "y": y, "z": points["Depth_m"].to_numpy()})
    means = key.groupby(["i", "j"])[["x", "y", "z"]].mean()
    return means[["x", "y"]].to_numpy(), means["z"].to_numpy()


def shore_points(mask: gpd.GeoDataFrame, spacing: float) -> np.ndarray:
    """Points every `spacing` metres along the boundary of the water mask, (n, 2)."""
    coords = []
    for boundary in mask.geometry.boundary.explode(index_parts=False):
        distances = np.arange(0, boundary.length, spacing)
        coords += [boundary.interpolate(d).coords[0] for d in distances]
    return np.array(coords).reshape(-1, 2)


def idw(xy: np.ndarray, z: np.ndarray, targets: np.ndarray, power: float, neighbours: int) -> np.ndarray:
    """Inverse distance weighting over the nearest `neighbours` points."""
    k = min(neighbours, len(z))
    distance, index = cKDTree(xy).query(targets, k=k)
    if k == 1:
        distance, index = distance[:, None], index[:, None]
    weights = 1 / (distance**power + 1e-12)
    return (weights * z[index]).sum(axis=1) / weights.sum(axis=1)


def grid_cells(mask: gpd.GeoDataFrame, pixel: float):
    """Cells of a `pixel` grid covering `mask`: (rows, cols, xy of their centres, shape, transform)."""
    minx, miny, maxx, maxy = mask.total_bounds
    width, height = int(np.ceil((maxx - minx) / pixel)), int(np.ceil((maxy - miny) / pixel))
    transform = from_origin(minx, maxy, pixel, pixel)
    inside = features.rasterize(
        ((g, 1) for g in mask.geometry), out_shape=(height, width), transform=transform, fill=0, dtype="uint8"
    ).astype(bool)
    rows, cols = np.nonzero(inside)
    centres = np.column_stack([minx + (cols + 0.5) * pixel, maxy - (rows + 0.5) * pixel])
    return rows, cols, centres, (height, width), transform


def interpolate(
    points: gpd.GeoDataFrame,
    mask: gpd.GeoDataFrame,
    pixel: float,
    method: str = "linear",
    shore_depth: float | None = None,
    max_distance: float | None = None,
    idw_power: float = 2.0,
    idw_neighbours: int = 12,
    thin_points: bool = True,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Grid the soundings inside the water mask.

    Parameters
    ----------
    points : geopandas.GeoDataFrame
        Soundings with ``Depth_m``, in the projected CRS of `mask`.
    mask : geopandas.GeoDataFrame
        Water polygons; cells outside are NaN.
    pixel : float
        Cell size (m).
    method : str
        ``idw``, ``linear``, ``cubic``, ``nearest`` (scipy griddata) or
        ``kriging`` (ordinary kriging, needs pykrige). Linear and cubic leave
        cells outside the soundings' convex hull to the nearest sounding.
    shore_depth : float or None
        Depth given to points along the mask boundary; None adds none (the
        edge of a mask is rarely where the depth is 0).
    max_distance : float or None
        Cells further than this from any sounding are NaN.
    thin_points : bool
        Average the soundings per cell first (see `thin`).

    Returns
    -------
    (numpy.ndarray, rasterio.Affine)
        Depth grid (rows from north to south) and its transform.
    """
    if thin_points:
        xy, z = thin(points, pixel)
    else:
        xy, z = np.column_stack([points.geometry.x, points.geometry.y]), points["Depth_m"].to_numpy()
    if shore_depth is not None:
        shore = shore_points(mask, spacing=pixel)
        xy_all = np.vstack([xy, shore])
        z_all = np.concatenate([z, np.full(len(shore), shore_depth)])
    else:
        xy_all, z_all = xy, z

    rows, cols, targets, shape, transform = grid_cells(mask, pixel)
    logging.info("Interpolating %d soundings (%d after thinning) on %d cells (%s)", len(points), len(z), len(targets), method)
    if max_distance is not None:
        distance, _ = cKDTree(xy).query(targets)
        keep = distance <= max_distance
        rows, cols, targets = rows[keep], cols[keep], targets[keep]

    if method == "idw":
        values = idw(xy_all, z_all, targets, idw_power, idw_neighbours)
    elif method in ("linear", "cubic", "nearest"):
        values = griddata(xy_all, z_all, targets, method=method)
        if method != "nearest":
            gaps = np.isnan(values)
            values[gaps] = griddata(xy_all, z_all, targets[gaps], method="nearest")
    elif method == "kriging":
        from pykrige.ok import OrdinaryKriging

        values, _ = OrdinaryKriging(xy_all[:, 0], xy_all[:, 1], z_all, variogram_model="spherical").execute(
            "points", targets[:, 0], targets[:, 1]
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    grid = np.full(shape, np.nan)
    grid[rows, cols] = values
    return grid, transform


def sample(grid: np.ndarray, transform, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Grid values at points (NaN outside the grid)."""
    col = np.floor((np.asarray(x) - transform.c) / transform.a).astype(int)
    row = np.floor((np.asarray(y) - transform.f) / transform.e).astype(int)
    inside = (row >= 0) & (row < grid.shape[0]) & (col >= 0) & (col < grid.shape[1])
    out = np.full(len(col), np.nan)
    out[inside] = grid[row[inside], col[inside]]
    return out


def cross_validate(
    points: gpd.GeoDataFrame, mask: gpd.GeoDataFrame, methods: list[str], group: str = "File", every: int = 5, **kwargs
) -> pd.DataFrame:
    """Error of each interpolation method on soundings left out.

    Every `every`-th value of the `group` column (a file, or a block of a
    few minutes of a transect) is left out, the rest interpolated, and the
    grid compared with the left-out soundings. Keyword arguments go to
    `interpolate`; with `max_distance`, soundings far from the kept ones are
    not predicted (``n`` counts the predicted ones).

    Returns ``RMSE_m``, ``Bias_m`` (grid - sounding), ``MAE_m`` and ``n`` per method.
    """
    groups = sorted(points[group].unique())
    held = points[group].isin(groups[::every])
    test = points[held]
    rows = []
    for method in methods:
        grid, transform = interpolate(points[~held], mask, method=method, **kwargs)
        predicted = sample(grid, transform, test.geometry.x, test.geometry.y)
        error = (predicted - test["Depth_m"].to_numpy())[np.isfinite(predicted)]
        rows.append({"Method": method, "RMSE_m": np.sqrt(np.mean(error**2)), "Bias_m": error.mean(),
                     "MAE_m": np.abs(error).mean(), "n": error.size})
    return pd.DataFrame(rows).set_index("Method")


def crossovers(points: gpd.GeoDataFrame, pixel: float, group: str = "Survey") -> pd.DataFrame:
    """Depth differences between surveys in the cells several of them sampled.

    Each `group` (e.g. a day, a file) is averaged per cell; for each cell
    sampled by two or more groups, returns the ``Range_m`` (max - min of the
    group means), ``n_groups``, the cell centre ``x``/``y`` and the groups'
    ``Depth_m`` mean. Without a tide correction this is mostly the tide.
    """
    x, y = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
    table = pd.DataFrame({"i": np.floor(x / pixel), "j": np.floor(y / pixel), "g": points[group].to_numpy(),
                          "z": points["Depth_m"].to_numpy()})
    per_group = table.groupby(["i", "j", "g"])["z"].mean()
    stats = per_group.groupby(level=["i", "j"]).agg(["max", "min", "mean", "count"])
    stats = stats[stats["count"] >= 2]
    out = pd.DataFrame({
        "x": (stats.index.get_level_values("i") + 0.5) * pixel,
        "y": (stats.index.get_level_values("j") + 0.5) * pixel,
        "Range_m": (stats["max"] - stats["min"]).to_numpy(),
        "Depth_m": stats["mean"].to_numpy(),
        "n_groups": stats["count"].to_numpy(),
    })
    return out


def write_geotiff(path: Path, grid: np.ndarray, transform, crs: str) -> None:
    """Save a grid as a single-band float32 GeoTIFF (NaN = no data), compressed."""
    with rasterio.open(
        path, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate", tiled=True,
    ) as dst:
        dst.write(grid.astype("float32"), 1)
