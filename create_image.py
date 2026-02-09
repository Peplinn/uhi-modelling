"""
create_image.py

This module gets the spectral indices using the specified city's shapefile.
"""
import geopandas as gpd
import json
import ee
import geemap

def get_spectral(
        country_code: str,
        city: str):

    country = gpd.read_file(f"data/shapefiles/gadm41_{country_code}_2.shp")

    city = country[country["NAME_1"] == city]

    city_geom = city.unary_union

    # Convert to GeoJSON format
    city_geojson = json.loads(gpd.GeoSeries([city_geom]).to_json())["features"][0][
        "geometry"
    ]

    city_aoi = ee.Geometry(city_geojson)

    # Creating the image and adding all the "spectral" bands

    image = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(city_aoi)
        .filterDate("2022-01-01", "2022-12-31") # Make this dynamic
        .median()
        .clip(city_aoi)
    )

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

    elevation = ee.Image("USGS/SRTMGL1_003").clip(city_aoi).rename("Elevation")

    landcover = ee.Image("ESA/WorldCover/v100/2020").clip(city_aoi).rename("LandCover")

    # Merging all the features:

    features = ndvi.addBands([ndbi, mndwi, savi, albedo, lst, elevation, landcover])

    # Create a mask where landcover equals 50 (Urban)
    urban_mask = landcover.eq(50)

    # Sample ONLY from the urban pixels
    points = features.updateMask(urban_mask).sample(
        region=city_aoi, 
        scale=30, 
        numPixels=200, 
        geometries=True, 
        seed=50
    )


    def add_latlon(feature):
        coords = feature.geometry().coordinates()
        lon = coords.get(0)
        lat = coords.get(1)
        return feature.set({"longitude": lon, "latitude": lat})


    # Map the function over the FeatureCollection to add the lat/lon properties
    points_with_latlon = points.map(add_latlon)

    geemap.ee_to_csv(points_with_latlon, filename=f"data/{country_code}_UHI_features.csv")