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
    
    sample_path = f"data/{country_code}_sample_points.csv"
    
    if os.path.exists(sample_path):
        print(f"Sample points already exist at {sample_path}.\n")
        return

    print(f"[{country_code}] Reading shapefile...")
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
    city_aoi = ee.Geometry(city_geojson).simplify(maxError=1000)

    print(f"[{country_code}] Sampling pixel locations...")

    # Annual Landsat values purely for stable pixel location sampling
    image = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(city_aoi)
        .filterDate(date[0], date[1])
        .median()
        .clip(city_aoi)
    )

    landcover = ee.Image("ESA/WorldCover/v100/2020").clip(city_aoi).rename("LandCover")
    elevation = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().clip(city_aoi).rename("Elevation")

    sampling_image = image.addBands([landcover, elevation])

    num_pixels = 3000 if country_code == "CAN" else 4000 if country_code == "FIN" else 1000

    points = sampling_image.sample(
        region=city_aoi,
        scale=30,
        numPixels=num_pixels,
        geometries=True,
        seed=50,
        tileScale=16
    )

    def add_latlon(feature):
        coords = feature.geometry().coordinates()
        return feature.set({"longitude": coords.get(0), "latitude": coords.get(1)})

    points_with_latlon = points.map(add_latlon)
    geemap.ee_to_csv(points_with_latlon, filename=f"data/{country_code}_raw_sample_points.csv")

    # Stratify to 100 urban / 100 rural
    df = pd.read_csv(f"data/{country_code}_raw_sample_points.csv")

    df = df[["latitude", "longitude", "LandCover", "Elevation"]]
    urban = df[df["LandCover"] == 50].sample(n=min(100, len(df[df["LandCover"] == 50])), random_state=42)
    rural = df[df["LandCover"] != 50].sample(n=min(100, len(df[df["LandCover"] != 50])), random_state=42)
    pixels = pd.concat([urban, rural]).reset_index(drop=True)

    print(f"[{country_code}] {len(urban)} urban, {len(rural)} rural pixels selected.")
    pixels.to_csv(sample_path, index=False)
    print(f"[{country_code}] Saved to {sample_path}.")

def get_viirs(
        country_code: str,
        date: tuple):

    viirs_path = f"data/{country_code}_viirs_features.csv"

    if os.path.exists(viirs_path):
        print(f"VIIRS data already exists at {viirs_path}.\n")
        return

    sample_path = f"data/{country_code}_sample_points.csv"
    if not os.path.exists(sample_path):
        print(f"Sample points not found for {country_code}. Run get_spectral first.")
        return

    pixels = pd.read_csv(sample_path)[["latitude", "longitude"]].drop_duplicates()

    features_fc = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([row["longitude"], row["latitude"]]),
            {"latitude": row["latitude"], "longitude": row["longitude"]}
        )
        for _, row in pixels.iterrows()
    ])

    year = pd.Timestamp(date[0]).year
    all_results = []

    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

        print(f"[{country_code}] Fetching VIIRS for month {month}...")

        # LST day from VNP21A1D
        lst_day = (
            ee.ImageCollection("NASA/VIIRS/002/VNP21A1D")
            .filterDate(start, end)
            .select("LST_1KM")
            .mean()
            .multiply(0.02)
            .rename("LST_day")
        )

        # LST night from VNP21A1N
        lst_night = (
            ee.ImageCollection("NASA/VIIRS/002/VNP21A1N")
            .filterDate(start, end)
            .select("LST_1KM")
            .mean()
            .multiply(0.02)
            .rename("LST_night")
        )

        # Spectral indices from VNP09GA
        vnp09 = (
            ee.ImageCollection("NASA/VIIRS/002/VNP09GA")
            .filterDate(start, end)
            .mean()
        )

        m3 = vnp09.select("M3")
        m4 = vnp09.select("M4")
        m5 = vnp09.select("M5")
        m7 = vnp09.select("M7")
        m10 = vnp09.select("M10")

        ndvi = m7.subtract(m5).divide(m7.add(m5)).rename("NDVI")
        ndbi = m10.subtract(m7).divide(m10.add(m7)).rename("NDBI")
        mndwi = m4.subtract(m10).divide(m4.add(m10)).rename("MNDWI")
        savi = m7.subtract(m5).divide(m7.add(m5).add(0.5)).multiply(1.5).rename("SAVI")
        albedo = m3.multiply(0.246) \
                   .add(m4.multiply(0.146)) \
                   .add(m5.multiply(0.191)) \
                   .add(m7.multiply(0.304)) \
                   .add(m10.multiply(0.113)) \
                   .rename("Albedo")

        combined = lst_day.addBands([lst_night, ndvi, ndbi, mndwi, savi, albedo])

        sampled = combined.sampleRegions(
            collection=features_fc,
            properties=["latitude", "longitude"],
            scale=500
        )

        month_data = sampled.getInfo()["features"]
        for f in month_data:
            props = f["properties"]
            props["month"] = month
            all_results.append(props)

        print(f"[{country_code}] Month {month} — {len(month_data)} points fetched.")

    pd.DataFrame(all_results).to_csv(viirs_path, index=False)
    print(f"[{country_code}] Saved {len(all_results)} rows to {viirs_path}.")