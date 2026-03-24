"""
fetch_data.py

This module fetches weather data from the Open-Meteo API
and provides fuzzy matching for city and country names.
"""
import os
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
import time

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(
    cache_session, 
    retries=5, 
    backoff_factor=2
)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"

today = datetime.today().strftime('%Y-%m-%d')

date_format = "%Y %B %d"


def fetch_data(lat: int,
               lng: int,
               date1: str,
               date2: str):
    """
    Fetches weather data from Open-Meteo API

    Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        variable (str): The weather variable to fetch
            ('temperature', 'rainfall', 'humidity').
        date1 (str): Start date in 'YYYY-MM-DD' format.
        date2 (str, optional): End date in 'YYYY-MM-DD' format.
            Defaults to today's date.

    Returns:
        pd.DataFrame: A pandas DataFrame with 'date'
            and the specified weather variable.

    Raises:
        ValueError: If an invalid weather variable is provided.
    """

    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": date1,
        "end_date": date2,
        "hourly": [
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "cloud_cover_low",
            "temperature_2m"
        ]
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    data = {"date": time_index}
    var_names = ["humidity", "precipitation", "wind_speed", "cloud_cover_low", "air_temperature"]

    for i, name in enumerate(var_names):
        data[name] = hourly.Variables(i).ValuesAsNumpy()

    df_full = pd.DataFrame(data)
    target_hours = [1, 13]

    df_filtered = df_full[
        (df_full['date'].dt.day == 1) & 
        (df_full['date'].dt.hour.isin(target_hours))
    ]

    return df_filtered.reset_index(drop=True)


def process_city_weather(
        country_code: str,
        date):

    csv_path = f"data/{country_code}_sample_points.csv"
    processed_path = f"data/{country_code}_Full_UHI_Data.csv"

    if os.path.exists(processed_path):
        print(f"Data has already been processed at {processed_path}.\n")
        return

    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    features_df = pd.read_csv(csv_path)

    all_weather_data = []

    # Iterate through each of the 200 sampled points
    no_rows = features_df.shape[0]

    if no_rows < 195:
        print(f"Less than 200 points ({no_rows}) sampled.")
        return

    for index, row in features_df.iterrows():
        lat = row['latitude']
        lng = row['longitude']

        time.sleep(3)

        print(f"Fetching weather for Point {index+1}/{no_rows} in {country_code}...")

        try:
            point_weather = fetch_data(lat, lng, date[0], date[1])

            print(f"Successfully fetched data for {lat}, {lng}, {date[0]}-{date[1]}")

            # Attach the GEE features (NDVI, LST, etc.) to this weather data
            # This links the spatial context to the temporal weather data
            for col in features_df.columns:
                point_weather[col] = row[col]

            all_weather_data.append(point_weather)
        except Exception as e:
            print(f"Error fetching data for {lat}, {lng}: {e}")

    # Combine all 200 points into one large "Master" dataset for the city
    final_city_df = pd.concat(all_weather_data, ignore_index=True)

    final_city_df.to_csv(f"data/{country_code}_Full_UHI_Data.csv", index=False)
    print(f"Saved complete dataset for {country_code} at data/{country_code}_Full_UHI_Data.csv")