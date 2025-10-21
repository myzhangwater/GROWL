import pandas as pd
from pathlib import Path
from shapely.geometry import Point
from shapely.validation import make_valid as shapely_make_valid
import geopandas as gpd
import numpy as np

def ensure_valid(gdf):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(lambda geom: shapely_make_valid(geom) if geom is not None else None)
    return gdf

def read_prepare_polys(path, id_col, buffer_m):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=4326)
    gdf = ensure_valid(gdf)

    gdf_m = gdf.to_crs(epsg=3857).copy()
    gdf_buff_m = gdf_m.copy()
    gdf_buff_m["geometry"] = gdf_buff_m.geometry.buffer(buffer_m)

    keep_cols = [id_col, "geometry"]
    return {
        "orig_4326": gdf,
        "orig_3857": gdf_m[keep_cols],
        "buff_3857": gdf_buff_m[keep_cols],
        "id_col": id_col,
        "buffer_m": buffer_m}

def match_points_to_layer(points_wgs84, layer, station_col, prefix):
    """
      - <prefix>_match_type: within / buffer / NaN
      - <prefix>_match_id
      - <prefix>_match_distance_m
        * within: 0
        * buffer: Point → Polygon boundary Shortest distance (m)
    """
    id_col = layer["id_col"]
    pts_m  = points_wgs84.to_crs(epsg=3857)
    poly_m = layer["orig_3857"]
    buff_m = layer["buff_3857"]

    poly_cent = poly_m[[id_col, "geometry"]].copy()
    poly_cent["centroid"] = poly_cent.geometry.centroid

    res = pd.DataFrame({station_col: points_wgs84[station_col].values})

    # ---------- within ----------
    within_best = pd.DataFrame(columns=[station_col, id_col, "centroid_dist_m"])
    within = gpd.sjoin(pts_m, poly_m[[id_col, "geometry"]], how="inner", predicate="within", rsuffix="_poly")
    if not within.empty:
        # Retain the first hit record for each site
        within_best = within.groupby(station_col, as_index=False).first()

        w = within_best[[station_col, id_col]].rename(columns={id_col: f"{prefix}_id_w"})
        w[f"{prefix}_dist_w"] = 0.0
        res = res.merge(w, on=station_col, how="left")

    # ---------- buffer ----------
    matched_stations = set(within_best[station_col]) if not within_best.empty else set()
    sub_pts = pts_m[~pts_m[station_col].isin(matched_stations)].copy()

    if not sub_pts.empty:
        buf_join = gpd.sjoin(sub_pts, buff_m[[id_col, "geometry"]], how="inner", predicate="within", rsuffix="_buf")
        if not buf_join.empty:
            buf_join = buf_join.merge(
                poly_m[[id_col, "geometry"]].rename(columns={"geometry": "geom_orig"}),
                on=id_col, how="left"
            )
            geom_orig_series = gpd.GeoSeries(buf_join["geom_orig"], crs=poly_m.crs)
            buf_join["boundary_dist_m"] = buf_join.geometry.distance(geom_orig_series)
            buf_best = buf_join.sort_values("boundary_dist_m").groupby(station_col, as_index=False).first()

            b = buf_best[[station_col, id_col, "boundary_dist_m"]].rename(
                columns={id_col: f"{prefix}_id_b", "boundary_dist_m": f"{prefix}_dist_b"}
            )
            res = res.merge(b, on=station_col, how="left")

    for c in [f"{prefix}_id_w", f"{prefix}_dist_w", f"{prefix}_id_b", f"{prefix}_dist_b"]:
        if c not in res.columns:
            res[c] = pd.NA

    res[f"{prefix}_match_type"] = np.select(
        [res[f"{prefix}_id_w"].notna(), res[f"{prefix}_id_b"].notna()],
        ["within", "buffer"],
        default=np.nan
    )

    res[f"{prefix}_match_id"] = res[f"{prefix}_id_w"].combine_first(res[f"{prefix}_id_b"])
    res[f"{prefix}_match_distance_m"] = res[f"{prefix}_dist_w"].combine_first(res[f"{prefix}_dist_b"])

    res.drop(columns=[f"{prefix}_id_w", f"{prefix}_dist_w", f"{prefix}_id_b", f"{prefix}_dist_b"], inplace=True)

    return res
