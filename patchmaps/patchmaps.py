import math
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


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_structure(
    fid: int,
    poly: Polygon,
    crs: str = "epsg:4326",
    working_width: int = 36,
    factor: int = 1,
    tramline: gpd.GeoDataFrame | None = None,
    use_pca: bool = False,
) -> gpd.GeoDataFrame:
    edge_length = working_width * factor

    input_crs = CRS.from_user_input(crs).name
    utm = CRS(input_crs)

    # If not a projected crs then try to find the correct one
    # If already projected just use it
    if not CRS(input_crs).is_projected:
        # find correct EPSG for calculation in meter
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
    # to utm (meters) TODO verify this.
    project = pyproj.Transformer.from_crs(crs, utm, always_xy=True).transform
    poly = transform(project, poly)

    if tramline is not None:
        tramline = tramline.to_crs(f"{utm}")
        p0 = np.array(
            tramline["geometry"][0].coords[0],
            dtype=np.float64,
        )  # First coordinate of permanent traffic laneb
        p1 = np.array(tramline["geometry"][0].coords[1], dtype=np.float64)
        direction = unit_vector(p1 - p0)
    elif use_pca:
        coords = np.array(poly.exterior.coords, dtype=np.float64)
        pca = PCA(n_components=2)
        _ = pca.fit(coords)
        direction = pca.components_[0]
    else:
        max_len = 0.0
        b = poly.exterior.coords
        linestrings = [LineString(b[k : k + 2]) for k in range(len(b) - 1)]
        longest: LineString | None = None
        for line in linestrings:
            if line.length >= max_len:
                longest = line
                max_len = line.length

        if longest is None:
            raise ValueError("Polygon must have at least one edge")

        direction = np.array(longest.coords[1]) - np.array(longest.coords[0])

    x_dir = np.array([1.0, 0.0], dtype=np.float64)
    # print(fid)
    angle = angle_between(x_dir, direction)
    angle_orig = angle
    cross = np.cross(x_dir, direction)
    if cross >= 0.0:
        angle = -angle

    rotated = shapely.affinity.rotate(
        poly,
        angle,
        origin=(poly.exterior.centroid.x, poly.exterior.centroid.y),
        use_radians=True,
    )
    x_diff = rotated.bounds[2] - rotated.bounds[0]
    y_diff = rotated.bounds[3] - rotated.bounds[1]

    # get the right dimension for layout
    dimension = x_diff / edge_length, y_diff / edge_length
    dimension_a = math.ceil(dimension[0])
    dimension_b = math.ceil(dimension[1])
    q1 = np.array([edge_length, 0], dtype=np.float64)
    q2 = np.array([0, edge_length], dtype=np.float64)
    # bottom left of grid
    so = np.array([rotated.bounds[0], rotated.bounds[1]], dtype=np.float64)

    # print(so)
    def compute_poly(i, j, rot):
        s = so + i * q1 + j * q2
        patch = Polygon([s, s + q1, s + (q1 + q2), s + q2])
        return shapely.affinity.rotate(
            patch,
            rot,
            origin=(poly.exterior.centroid.x, poly.exterior.centroid.y),
            use_radians=True,
        )
        # return shapely.affinity.rotate(patch, rot, origin=(0.0, 0.0), use_radians=True)
        # return shapely.affinity.rotate(patch, a, origin='center', use_radians=True)

    polies = [
        compute_poly(i, j, -angle) for i, j in product(range(dimension_a), range(dimension_b))
    ]
    data = gpd.GeoDataFrame({"geometry": polies})
    data.crs = f"{utm}"

    grid_poly = data.iloc[0]["geometry"]
    x1 = np.array(grid_poly.exterior.coords[0], dtype=np.float64)
    x2 = np.array(grid_poly.exterior.coords[1], dtype=np.float64)
    vec = x2 - x1
    a = angle_between(vec, x_dir)
    vec_cross = np.cross(vec, direction)
    ab = angle_between(vec, direction)
    if (
        not math.isclose(a, angle_orig)
        and not math.isclose(vec_cross, 0.0)
        and not math.isclose(ab, 0.0)
    ):
        print(f"""WRONG {fid}, {a}, {ab}, {angle}, {angle_orig}, {vec_cross}, {cross}""")

    # clip to boundary
    patches_within = data.clip(poly, keep_geom_type=True)
    sum_area = patches_within["geometry"].apply(lambda geom: geom.area).sum()

    if not math.isclose(sum_area, poly.area):
        print(f"""Different area for {fid}: {sum_area} {poly.area}""")

    patches_within = patches_within.to_crs(crs)

    return patches_within
