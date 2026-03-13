"""
create_image.py

This module gets the spectral indices using the specified city's shapefile.
"""
import geopandas as gpd
import json
import ee
import geemap
import os
import pandas as pd

def get_spectral(
        country_code: str,
        city_name: str,
        date: tuple):
    
    spectral_path = f"data/{country_code}_spectral_features.csv"
    
    if os.path.exists(spectral_path):
        print(f"Data has already been processed at {spectral_path}.\n")
        return

    country = gpd.read_file(f"data/shapefiles/gadm41_{country_code}_2.shp")

    if country_code == "CAN":
        city = country[country["NAME_2"].isin(["Toronto", "Peel", "York", "Durham", "Halton", "Hamilton", "Waterloo", "Niagara"])]
    elif country_code == "IRN":
        city = country[country["NAME_2"].isin(["Theran", "Rey", "Shemiranat"])]
    elif country_code == "FIN":
        city = country[(country["NAME_1"] == "Southern Finland") & (country["NAME_2"] == "Uusimaa")]
    else:
        city = country[country["NAME_1"] == city_name]

    city_geom = city.unary_union
    city_geojson = json.loads(gpd.GeoSeries([city_geom]).to_json())["features"][0]["geometry"]
    city_aoi = ee.Geometry(city_geojson)

    image = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(city_aoi)
        .filterDate(date[0], date[1])
        .median()
        .clip(city_aoi)
    )

    ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    mndwi = image.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI")
    savi = image.expression(
        "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
        {"NIR": image.select("SR_B5"), "RED": image.select("SR_B4"), "L": 0.5}
    ).rename("SAVI")
    elevation = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().clip(city_aoi).rename("Elevation")
    landcover = ee.Image("ESA/WorldCover/v100/2020").clip(city_aoi).rename("LandCover")

    features = ndbi.addBands([mndwi, savi, elevation, landcover])

    num_pixels = 3000 if country_code in ["CAN", "FIN"] else 1000

    points = features.sample(
        region=city_aoi,
        scale=30,
        numPixels=num_pixels,
        geometries=True,
        seed=50
    )

    def add_latlon(feature):
        coords = feature.geometry().coordinates()
        return feature.set({"longitude": coords.get(0), "latitude": coords.get(1)})

    points_with_latlon = points.map(add_latlon)

    geemap.ee_to_csv(points_with_latlon, filename=f"data/{country_code}_spectral_features.csv")

    df = pd.read_csv(f"data/{country_code}_spectral_features.csv")
    urban = df[df["LandCover"] == 50].sample(n=min(100, len(df[df["LandCover"] == 50])), random_state=42)
    rural = df[df["LandCover"] != 50].sample(n=min(100, len(df[df["LandCover"] != 50])), random_state=42)
    pd.concat([urban, rural]).reset_index(drop=True).to_csv(f"data/{country_code}_spectral_features.csv", index=False)
    print(f"Saved {country_code} — {len(urban)} urban, {len(rural)} rural pixels.")


def get_modis(
        country_code: str,
        date: tuple):

    year = pd.Timestamp(date[0]).year
    modis_path = f"data/{country_code}_modis_features.csv"

    if os.path.exists(modis_path):
        print(f"MODIS data already exists at {modis_path}.\n")
        return

    spectral_path = f"data/{country_code}_spectral_features.csv"
    if not os.path.exists(spectral_path):
        print(f"Spectral features not found for {country_code}. Run get_spectral first.")
        return

    pixels = pd.read_csv(spectral_path)[["latitude", "longitude"]].drop_duplicates()

    results = []

    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

        print(f"[{country_code}] Fetching MODIS for month {month}...")

        features = [
            ee.Feature(
                ee.Geometry.Point([row["longitude"], row["latitude"]]),
                {"latitude": row["latitude"], "longitude": row["longitude"]}
            )
            for _, row in pixels.iterrows()
        ]
        fc = ee.FeatureCollection(features)

        # LST daytime and nighttime from Terra MOD11A1
        lst_day = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterDate(start, end)
            .select("LST_Day_1km")
            .mean()
            .multiply(0.02)  # scale factor
            .rename("LST_day")
        )

        lst_night = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterDate(start, end)
            .select("LST_Night_1km")
            .mean()
            .multiply(0.02)
            .rename("LST_night")
        )

        # NDVI from MOD13A3
        ndvi = (
            ee.ImageCollection("MODIS/061/MOD13A3")
            .filterDate(start, end)
            .select("1_km_monthly_NDVI")
            .mean()
            .multiply(0.0001)  # scale factor
            .rename("NDVI")
        )

        # Albedo from MCD43A3
        albedo = (
            ee.ImageCollection("MODIS/061/MCD43A3")
            .filterDate(start, end)
            .select("Albedo_WSA_shortwave")
            .mean()
            .multiply(0.001)  # scale factor
            .rename("Albedo")
        )

        combined = lst_day.addBands([lst_night, ndvi, albedo])

        sampled = combined.sampleRegions(
            collection=fc,
            properties=["latitude", "longitude"],
            scale=1000
        )

        month_data = sampled.getInfo()["features"]
        for f in month_data:
            props = f["properties"]
            props["month"] = month
            results.append(props)

    pd.DataFrame(results).to_csv(modis_path, index=False)
    print(f"Saved MODIS features for {country_code}.")