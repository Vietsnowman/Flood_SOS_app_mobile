import requests
import pandas as pd

def fetch_open_meteo_daily(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_hours",
            "wind_speed_10m_max"
        ],
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "precipitation",
            "wind_speed_10m"
        ],
        "timezone": "Asia/Bangkok"
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo Error {r.status_code}: {r.text[:300]}")

    data = r.json()

    daily = pd.DataFrame(data["daily"])
    daily.rename(columns={"time": "date_local"}, inplace=True)
    daily["date_local"] = pd.to_datetime(daily["date_local"])

    hourly = pd.DataFrame(data["hourly"])
    hourly.rename(columns={"time": "time_local"}, inplace=True)
    hourly["time_local"] = pd.to_datetime(hourly["time_local"])
    hourly["date_local"] = hourly["time_local"].dt.floor("D")

    agg = hourly.groupby("date_local").agg(
        temperature_2m_mean=("temperature_2m", "mean"),
        relative_humidity_2m_mean=("relative_humidity_2m", "mean"),
        cloud_cover_mean=("cloud_cover", "mean"),
        wind_speed_10m_mean=("wind_speed_10m", "mean"),
        precipitation_max_1h=("precipitation", "max")
    ).reset_index()

    out = daily.merge(agg, on="date_local", how="left")
    return out