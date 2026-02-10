"""Provide functionality for patchmaps."""

import logging
import math
from dataclasses import dataclass
from itertools import product

import geopandas as gpd
import numpy as np
import pyproj
import shapely.affinity
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely import LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class StructureOptions:
    """Configuration for `get_structure`."""

    crs: str = "epsg:4326"
    working_width: int = 36
    factor: int = 1
    tramline: gpd.GeoDataFrame | None = None
    use_pca: bool = False


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Execute unit vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Execute angle between."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def _get_projected_crs(poly: Polygon, crs: str) -> CRS:
    input_crs = CRS.from_user_input(crs).name
    utm = CRS(input_crs)
    if not CRS(input_crs).is_projected:
        utm_crs_list = query_utm_crs_info(
            datum_name=input_crs,
            area_of_interest=AreaOfInterest(
                west_lon_degree=poly.bounds[0],
                south_lat_degree=poly.bounds[1],
                east_lon_degree=poly.bounds[2],
                north_lat_degree=poly.bounds[3],
            ),
        )
        utm = CRS.from_epsg(utm_crs_list[0].code)
    return utm


def _project_polygon(poly: Polygon, crs: str, projected_crs: CRS) -> Polygon:
    project = pyproj.Transformer.from_crs(crs, projected_crs, always_xy=True).transform
    return transform(project, poly)


def _longest_edge_direction(poly: Polygon) -> np.ndarray:
    max_len = 0.0
    border = poly.exterior.coords
    linestrings = [LineString(border[k : k + 2]) for k in range(len(border) - 1)]
    longest: LineString | None = None
    for line in linestrings:
        if line.length >= max_len:
            longest = line
            max_len = line.length
    if longest is None:
        msg = "Polygon must have at least one edge"
        raise ValueError(msg)
    return np.array(longest.coords[1]) - np.array(longest.coords[0])


def _get_direction(poly: Polygon, projected_crs: CRS, options: StructureOptions) -> np.ndarray:
    tramline = options.tramline
    if tramline is not None:
        tramline = tramline.to_crs(f"{projected_crs}")
        p0 = np.array(
            tramline["geometry"][0].coords[0],
            dtype=np.float64,
        )  # First coordinate of permanent traffic laneb
        p1 = np.array(tramline["geometry"][0].coords[1], dtype=np.float64)
        return unit_vector(p1 - p0)
    if options.use_pca:
        coords = np.array(poly.exterior.coords, dtype=np.float64)
        pca = PCA(n_components=2)
        _ = pca.fit(coords)
        return pca.components_[0]
    return _longest_edge_direction(poly)


def _rotation_values(direction: np.ndarray) -> tuple[float, float, np.float64]:
    x_dir = np.array([1.0, 0.0], dtype=np.float64)
    angle = angle_between(x_dir, direction)
    angle_orig = angle
    cross = np.cross(x_dir, direction)
    if cross >= 0.0:
        angle = -angle
    return angle, angle_orig, cross


def _build_grid(
    poly: Polygon,
    rotated: Polygon,
    angle: float,
    edge_length: int,
) -> gpd.GeoDataFrame:
    x_diff = rotated.bounds[2] - rotated.bounds[0]
    y_diff = rotated.bounds[3] - rotated.bounds[1]
    dimension = x_diff / edge_length, y_diff / edge_length
    dimension_a = math.ceil(dimension[0])
    dimension_b = math.ceil(dimension[1])
    q1 = np.array([edge_length, 0], dtype=np.float64)
    q2 = np.array([0, edge_length], dtype=np.float64)
    origin = np.array([rotated.bounds[0], rotated.bounds[1]], dtype=np.float64)

    def compute_poly(i: int, j: int, rot: float) -> Polygon:
        s = origin + i * q1 + j * q2
        patch = Polygon([s, s + q1, s + (q1 + q2), s + q2])
        return shapely.affinity.rotate(
            patch,
            rot,
            origin=(poly.exterior.centroid.x, poly.exterior.centroid.y),
            use_radians=True,
        )

    polies = [compute_poly(i, j, -angle) for i, j in product(range(dimension_a), range(dimension_b))]
    return gpd.GeoDataFrame({"geometry": polies})


def _log_orientation_warning(
    data: gpd.GeoDataFrame,
    fid: int | str,
    direction: np.ndarray,
    rotation_values: tuple[float, float, np.float64],
) -> None:
    angle, angle_orig, cross = rotation_values
    x_dir = np.array([1.0, 0.0], dtype=np.float64)
    grid_poly = data.iloc[0]["geometry"]
    x1 = np.array(grid_poly.exterior.coords[0], dtype=np.float64)
    x2 = np.array(grid_poly.exterior.coords[1], dtype=np.float64)
    vec = x2 - x1
    a = angle_between(vec, x_dir)
    vec_cross = np.cross(vec, direction)
    ab = angle_between(vec, direction)
    if not math.isclose(a, angle_orig) and not math.isclose(vec_cross, 0.0) and not math.isclose(ab, 0.0):
        logger.warning(
            "WRONG %s, %s, %s, %s, %s, %s, %s",
            fid,
            a,
            ab,
            angle,
            angle_orig,
            vec_cross,
            cross,
        )


def get_structure(
    fid: int | str,
    poly: Polygon,
    options: StructureOptions | None = None,
) -> gpd.GeoDataFrame:
    """Return structure."""
    structure_options = options or StructureOptions()
    edge_length = structure_options.working_width * structure_options.factor
    utm = _get_projected_crs(poly, structure_options.crs)
    poly = _project_polygon(poly, structure_options.crs, utm)
    direction = _get_direction(poly, utm, structure_options)

    angle, angle_orig, cross = _rotation_values(direction)

    rotated = shapely.affinity.rotate(
        poly,
        angle,
        origin=(poly.exterior.centroid.x, poly.exterior.centroid.y),
        use_radians=True,
    )
    data = _build_grid(poly, rotated, angle, edge_length)
    data.crs = f"{utm}"
    _log_orientation_warning(data, fid, direction, (angle, angle_orig, cross))

    # clip to boundary
    patches_within = data.clip(poly, keep_geom_type=True)
    sum_area = patches_within["geometry"].apply(lambda geom: geom.area).sum()

    if not math.isclose(sum_area, poly.area):
        logger.warning("Different area for %s: %s %s", fid, sum_area, poly.area)

    return patches_within.to_crs(structure_options.crs)
