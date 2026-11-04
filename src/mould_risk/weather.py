"""Hourly outdoor temperature and humidity per IDEAL location, from the Open-Meteo archive (ERA5)."""

import json
import urllib.parse
import urllib.request

import pandas as pd

from mould_risk.paths import PROCESSED

# weather centres named in metadata/location.csv
CENTRES = {
    "Edinburgh": (55.953, -3.189),
    "WestLothian": (55.883, -3.516),
    "Midlothian": (55.832, -3.226),
    "EastLothian": (56.058, -2.717),
    "Fife": (56.111, -3.167),
}
START, END = "2016-08-01", "2018-07-01"
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(locationid, lat, lon):
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
        "hourly": "temperature_2m,relative_humidity_2m", "timezone": "GMT",
    })
    with urllib.request.urlopen(f"{URL}?{query}", timeout=60) as r:
        hourly = json.load(r)["hourly"]
    return pd.DataFrame({
        "locationid": locationid,
        "ts": pd.to_datetime(hourly["time"]),
        "t_out": hourly["temperature_2m"],
        "rh_out": hourly["relative_humidity_2m"],
    })


def build():
    frames = [fetch(loc, lat, lon) for loc, (lat, lon) in CENTRES.items()]
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = out["ts"].astype("datetime64[s]")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED / "weather_hourly.parquet", index=False)
    return out


if __name__ == "__main__":
    w = build()
    print(w.groupby("locationid")[["t_out", "rh_out"]].describe().round(1).T)
