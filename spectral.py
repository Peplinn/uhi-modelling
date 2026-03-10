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
    
    print("Read shapefile...")

    country = gpd.read_file(f"data/shapefiles/gadm41_{country_code}_2.shp")

    if country_code == "CAN":
        urban_districts = ["Toronto", "Peel", "York", "Durham", "Halton", "Hamilton", "Waterloo", "Niagara"]
        city = country[country["NAME_2"].isin(urban_districts)]
    elif country_code == "IRN":
        city = country[country["NAME_2"].isin(["Theran", "Rey", "Shemiranat"])]
    elif country_code == "FIN":
        city = country[(country["NAME_1"] == "Southern Finland") & (country["NAME_2"] == "Uusimaa")]
    else:
        city = country[country["NAME_1"] == city_name]

    city_geom = city.unary_union

    # Convert to GeoJSON format
    city_geojson = json.loads(gpd.GeoSeries([city_geom]).to_json())["features"][0][
        "geometry"
    ]

    city_aoi = ee.Geometry(city_geojson).simplify(maxError=1000)

    # Creating the image and adding all the "spectral" bands
    print("Creating image from date range within shapefile bounds...")

    image = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(city_aoi)
        .filterDate(date[0], date[1]) # Make this dynamic
        .median()
        .clip(city_aoi)
    )

    print("Image created.")

    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

    # 
    lst = image.select("ST_B10").multiply(0.00341802).add(149).rename("LST")

    ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")

    mndwi = image.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI")

    savi = image.expression(
        "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
        {
            "NIR": image.select("SR_B5"),
            "RED": image.select("SR_B4"),
            "L": 0.5,  # soil brightness correction factor
        },
    ).rename("SAVI")

    albedo = image.expression(
        "0.356 * B2 + 0.130 * B4 + 0.373 * B5 + 0.085 * B6 + 0.072 * B7 - 0.0018",
        {
            "B2": image.select("SR_B2"),
            "B4": image.select("SR_B4"),
            "B5": image.select("SR_B5"),
            "B6": image.select("SR_B6"),
            "B7": image.select("SR_B7"),
        },
    ).rename("Albedo")

    elevation = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().clip(city_aoi).rename("Elevation")

    landcover = ee.Image("ESA/WorldCover/v100/2020").clip(city_aoi).rename("LandCover")

    # Merging all the features:
    print("Merging all the features...")

    features = ndvi.addBands([ndbi, mndwi, savi, albedo, lst, elevation, landcover])

    print("Features merged.")

    # Create a mask where landcover equals 50 (Urban)
    # urban_mask = landcover.eq(50)

    # Sample ONLY from the urban pixels
    # points = features.updateMask(urban_mask).sample(
    

    num_pixels = 3000 if country_code in ["CAN", "FIN"] else 1000

    num_pixels = 4000 if country_code == "FIN" else 1000

    print(f"Sampling {num_pixels} points...")

    points = features.sample(
        region=city_aoi,
        scale=30,
        numPixels=num_pixels,
        geometries=True,
        seed=50
    )

    print("Sampling done.")


    def add_latlon(feature):
        coords = feature.geometry().coordinates()
        lon = coords.get(0)
        lat = coords.get(1)
        return feature.set({"longitude": lon, "latitude": lat})

    print("Adding lat/lon for all the points")

    # Map the function over the FeatureCollection to add the lat/lon properties
    points_with_latlon = points.map(add_latlon)

    print("Done adding coordinates")

    geemap.ee_to_csv(points_with_latlon, filename=f"data/{country_code}_spectral_features.csv")

    print("Done with EE.")

    df = pd.read_csv(f"data/{country_code}_spectral_features.csv")
    print("Sampling Urban...")
    urban = df[df["LandCover"] == 50].sample(n=min(100, len(df[df["LandCover"] == 50])), random_state=42)

    print("Sampling Rural...")
    rural = df[df["LandCover"] != 50].sample(n=min(100, len(df[df["LandCover"] != 50])), random_state=42)

    print("Concatenating and saving...")
    pd.concat([urban, rural]).reset_index(drop=True).to_csv(f"data/{country_code}_spectral_features.csv", index=False)
    print(f"Done. Saved at \"data/{country_code}_spectral_features.csv\"")