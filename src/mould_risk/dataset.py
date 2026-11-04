"""Build the room-day forecasting table: targets, baselines and features for horizons 1-3 days."""

import zlib

import numpy as np
import pandas as pd

from mould_risk.paths import META, PROCESSED
from mould_risk.physics import SURFACE_RH_LIMIT, surface_rh, surface_temperature, vapour_pressure, vtt_index

ROOM_TYPES = ["bedroom", "livingroom", "kitchen", "bathroom", "kitchenliving", "diningroom", "study"]
HORIZONS = (1, 2, 3)
F_RSI = 0.75
MIN_HOURS = 20
MIN_READINGS = 150
HIGH_RISK_HOURS = 6
# Assumed forecast error sd per horizon (days): outdoor temperature in C, outdoor RH in %.
# Archived 2016-18 forecasts are not available, so forecasts are actuals plus this error.
FORECAST_SD = {1: (1.0, 5.0), 2: (1.5, 7.0), 3: (2.0, 9.0)}
ERROR_ORIGIN = pd.Timestamp("2015-01-01")
ERROR_DAYS = 2000
BUILD_ERA_ORDER = [
    "Before 1850", "1850-1899", "1900-1918", "1919-1930", "1931-1944", "1945-1964",
    "1965-1980", "1981-1990", "1991-1995", "1996-2001", "2002 or later",
]


def read_meta(name):
    return pd.read_csv(META / f"{name}.csv", encoding="utf-8-sig", low_memory=False)


def flagged_rooms():
    """Rooms whose room sensorbox is flagged as giving inaccurate temperature or humidity."""
    box = read_meta("sensorbox")
    box = box[box["sensorbox_type"] == "room"]
    bad = box[(box["temperatureInaccuracy"] == 1) | (box["humidityInaccuracy"] == 1)]
    return set(bad["roomid"])


def room_table():
    rooms = read_meta("room")
    homes = read_meta("home")
    rooms = rooms[rooms["type"].isin(ROOM_TYPES) & (pd.to_numeric(rooms["externalwalls"], errors="coerce") > 0)]
    rooms = rooms[~rooms["roomid"].isin(flagged_rooms())]
    homes = homes.assign(build_era_ord=homes["build_era"].map({e: i for i, e in enumerate(BUILD_ERA_ORDER)}))
    cols = ["homeid", "location", "hometype", "residents", "build_era_ord", "equivalised_income"]
    return rooms.merge(homes[cols], on="homeid", how="inner")


def day_matrices(hourly: pd.DataFrame):
    """Reshape one room's hourly series to (days, 24) arrays, NaN where an hour is not usable."""
    h = hourly.set_index("ts").sort_index()
    start = h.index.min().floor("D")
    end = h.index.max().floor("D") + pd.Timedelta(hours=23)
    h = h.reindex(pd.date_range(start, end, freq="h"))
    days = pd.date_range(start, end.floor("D"), freq="D")
    shape = (len(days), 24)
    return days, {c: h[c].to_numpy(float).reshape(shape) for c in ("t_in", "vp_in", "t_out", "rh_out")}


def fav_hours(t_in, vp_in, t_out, f_rsi=F_RSI):
    """Hours per day with surface RH >= 80 %, scaled to 24 h; NaN if fewer than MIN_HOURS usable."""
    rh_s = surface_rh(t_in, vp_in, t_out, f_rsi)
    valid = ~np.isnan(rh_s)
    n = valid.sum(axis=1)
    count = np.where(valid, rh_s >= SURFACE_RH_LIMIT, False).sum(axis=1)
    return np.where(n >= MIN_HOURS, count * 24.0 / np.maximum(n, 1), np.nan)


def forecast_error(location, h, dates, scale=1.0):
    """Daily forecast errors (t_out, rh_out), identical for every room sharing a location."""
    rng = np.random.default_rng([zlib.crc32(str(location).encode()), h])
    sd_t, sd_rh = FORECAST_SD[h]
    e = rng.standard_normal((ERROR_DAYS, 2)) * np.array([sd_t, sd_rh]) * scale
    i = (pd.DatetimeIndex(dates) - ERROR_ORIGIN).days
    return e[i, 0], e[i, 1]


def room_rows(hourly: pd.DataFrame, f_rsi=F_RSI, location=0, error_scale=1.0):
    days, m = day_matrices(hourly)
    fav = fav_hours(m["t_in"], m["vp_in"], m["t_out"], f_rsi)
    t_si = surface_temperature(m["t_in"], m["t_out"], f_rsi).ravel()
    vtt = vtt_index(t_si, surface_rh(m["t_in"], m["vp_in"], m["t_out"], f_rsi).ravel()).reshape(-1, 24)[:, -1]
    rh_in = 100.0 * m["vp_in"] / vapour_pressure(m["t_in"], 100.0)
    with np.errstate(all="ignore"):
        daily = {
            "t_in_mean": np.nanmean(m["t_in"], axis=1),
            "t_in_min": np.nanmin(m["t_in"], axis=1),
            "rh_in_mean": np.nanmean(rh_in, axis=1),
            "rh_in_max": np.nanmax(rh_in, axis=1),
            "vp_excess": np.nanmean(m["vp_in"] - vapour_pressure(m["t_out"], m["rh_out"]), axis=1),
            "t_out_mean": np.nanmean(m["t_out"], axis=1),
            "t_out_min": np.nanmin(m["t_out"], axis=1),
            "rh_out_mean": np.nanmean(m["rh_out"], axis=1),
        }
    fav_s = pd.Series(fav)
    rows = []
    for h in HORIZONS:
        n = len(days) - h
        if n <= 7:
            continue
        idx = np.arange(7, n)
        target = fav[idx + h]
        physics = fav_hours(m["t_in"][idx], m["vp_in"][idx], m["t_out"][idx + h], f_rsi)
        e_t, e_rh = forecast_error(location, h, days[idx + h], error_scale)
        physics_fc = fav_hours(m["t_in"][idx], m["vp_in"][idx], m["t_out"][idx + h] + e_t[:, None], f_rsi)
        frame = pd.DataFrame({
            "date": days[idx],
            "horizon": h,
            "target": target,
            "persistence": fav[idx],
            "physics_persistence": physics,
            "physics_forecast": physics_fc,
            "fav_lag1": fav[idx - 1],
            "fav_lag2": fav[idx - 2],
            "fav_mean7": fav_s.rolling(7, min_periods=5).mean().to_numpy()[idx],
            "vtt_surface": vtt[idx],
            **{k: v[idx] for k, v in daily.items()},
            "t_out_mean_target": daily["t_out_mean"][idx + h],
            "t_out_min_target": daily["t_out_min"][idx + h],
            "rh_out_mean_target": daily["rh_out_mean"][idx + h],
        })
        frame["t_out_mean_target_fc"] = frame["t_out_mean_target"] + e_t
        frame["t_out_min_target_fc"] = frame["t_out_min_target"] + e_t
        frame["rh_out_mean_target_fc"] = (frame["rh_out_mean_target"] + e_rh).clip(0, 100)
        frame["t_out_change"] = frame["t_out_mean_target"] - frame["t_out_mean"]
        frame["t_out_change_fc"] = frame["t_out_mean_target_fc"] - frame["t_out_mean"]
        target_day = days[idx + h]
        frame["doy_sin"] = np.sin(2 * np.pi * target_day.day_of_year / 365.25)
        frame["doy_cos"] = np.cos(2 * np.pi * target_day.day_of_year / 365.25)
        frame["weekend_target"] = (target_day.day_of_week >= 5).astype(int)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["target", "persistence", "physics_persistence"])


def hourly_with_weather(room_hourly, weather_hourly, rooms):
    ok = (room_hourly["temperature_n"] >= MIN_READINGS) & (room_hourly["humidity_n"] >= MIN_READINGS)
    rh = room_hourly[ok & room_hourly["roomid"].isin(rooms["roomid"])]
    rh = rh.merge(rooms[["roomid", "location"]], on="roomid")
    rh = rh.merge(weather_hourly, left_on=["location", "ts"], right_on=["locationid", "ts"], how="left")
    rh["vp_in"] = vapour_pressure(rh["t_in"], rh["rh_in"])
    return rh[["homeid", "roomid", "location", "ts", "t_in", "vp_in", "t_out", "rh_out"]]


ROOM_FEATURES = ["type", "storey", "externalwalls", "windowsopen", "radiators", "clothesdrying",
                 "hometype", "residents", "build_era_ord"]


def build(f_rsi=F_RSI, error_scale=1.0):
    rooms = room_table()
    hourly = hourly_with_weather(
        pd.read_parquet(PROCESSED / "room_hourly.parquet"), pd.read_parquet(PROCESSED / "weather_hourly.parquet"), rooms
    )
    parts = []
    for roomid, g in hourly.groupby("roomid"):
        rows = room_rows(g, f_rsi, g["location"].iloc[0], error_scale)
        if len(rows):
            rows.insert(0, "roomid", roomid)
            parts.append(rows)
    data = pd.concat(parts, ignore_index=True)
    data = data.merge(rooms[["roomid", "homeid", "equivalised_income", *ROOM_FEATURES]], on="roomid")
    for c in ("type", "clothesdrying", "hometype"):
        data[c] = data[c].astype("category")
    return data


if __name__ == "__main__":
    data = build()
    data.to_parquet(PROCESSED / "room_days.parquet", index=False)
    print(f"{len(data):,} rows, {data['roomid'].nunique()} rooms, {data['homeid'].nunique()} homes")
