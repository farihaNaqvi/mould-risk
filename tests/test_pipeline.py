import gzip
import zipfile

import numpy as np
import pandas as pd
import pytest

from mould_risk import ingest
from mould_risk.dataset import FORECAST_SD, fav_hours, forecast_error, room_rows
from mould_risk.evaluate import FEATURES, metrics
from mould_risk.physics import vapour_pressure


def test_parse_name_reads_ids_and_ignores_other_sensors():
    meta = ingest.parse_name("sensordata/home86_kitchenliving932_sensor3503_room_humidity.csv.gz")
    assert meta == {"homeid": 86, "roomtype": "kitchenliving", "roomid": 932, "sensorid": 3503, "kind": "humidity"}
    assert ingest.parse_name("home86_study932_sensor3502_room_light.csv.gz") is None
    assert ingest.parse_name("home96_kitchen999_sensor9110_electric-appliance_kettle.csv.gz") is None


@pytest.mark.parametrize("header", [b"", b"timestamp,value\n"])
def test_member_is_reduced_to_hourly_means_in_real_units(tmp_path, header):
    body = b"".join(f"2017-01-01 00:{m:02d}:00,{200 + m}\n".encode() for m in range(60))
    body += b"2017-01-01 01:00:12,300\n"
    path = tmp_path / "room.zip"
    name = "home1_bedroom10_sensor5_room_temperature.csv.gz"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, gzip.compress(header + body))
    out = ingest.process_member((path, name))
    assert list(out["n"]) == [60, 1]
    assert out["value"].iloc[0] == pytest.approx(22.95)
    assert out["value"].iloc[1] == pytest.approx(30.0)
    assert set(out["roomid"]) == {10}


def test_fav_hours_scales_partial_days_and_drops_sparse_ones():
    t_in = np.full((2, 24), 18.0)
    vp = np.full((2, 24), vapour_pressure(18.0, 70.0))
    t_out = np.full((2, 24), 0.0)
    t_in[0, :2] = np.nan
    t_in[1, :10] = np.nan
    out = fav_hours(t_in, vp, t_out)
    assert out[0] == pytest.approx(24.0)
    assert np.isnan(out[1])


def synthetic_room(days=30, cold_from=20):
    ts = pd.date_range("2017-01-01", periods=days * 24, freq="h")
    t_out = np.where(np.arange(len(ts)) >= cold_from * 24, -2.0, 10.0)
    return pd.DataFrame({
        "ts": ts, "t_in": 18.0, "vp_in": vapour_pressure(18.0, 62.0), "t_out": t_out, "rh_out": 85.0,
    })


def test_physics_persistence_sees_the_cold_spell_that_persistence_misses():
    rows = room_rows(synthetic_room())
    h3 = rows[rows["horizon"] == 3].set_index("date")
    day = pd.Timestamp("2017-01-18")
    assert h3.loc[day, "persistence"] == 0
    assert h3.loc[day, "target"] == 24
    assert h3.loc[day, "physics_persistence"] == 24
    assert set(rows["horizon"]) == {1, 2, 3}


def test_metrics_count_onsets():
    df = pd.DataFrame({"target": [10, 10, 0, 0], "persistence": [0, 10, 0, 10], "m": [8, 0, 0, 7]})
    out = metrics(df, "m")
    assert out["onset_n"] == 1 and out["onset_recall"] == 1.0
    assert out["precision"] == 0.5 and out["recall"] == 0.5


def test_forecast_error_is_shared_by_location_and_sized_by_horizon():
    dates = pd.date_range("2016-01-01", periods=600, freq="D")
    a_t, _ = forecast_error("loc1", 1, dates)
    b_t, _ = forecast_error("loc1", 1, dates)
    c_t, _ = forecast_error("loc2", 1, dates)
    h3_t, _ = forecast_error("loc1", 3, dates)
    assert np.array_equal(a_t, b_t) and not np.array_equal(a_t, c_t)
    assert np.std(a_t) == pytest.approx(FORECAST_SD[1][0], rel=0.15)
    assert np.std(h3_t) == pytest.approx(FORECAST_SD[3][0], rel=0.15)
    assert np.all(forecast_error("loc1", 1, dates, scale=0.0)[0] == 0)


def test_forecast_rows_carry_noisy_weather_and_zero_error_matches_oracle():
    rows = room_rows(synthetic_room(), location="loc1")
    assert not np.allclose(rows["t_out_mean_target_fc"], rows["t_out_mean_target"])
    exact = room_rows(synthetic_room(), location="loc1", error_scale=0.0)
    assert np.allclose(exact["physics_forecast"], exact["physics_persistence"])
    assert np.allclose(exact["t_out_change_fc"], exact["t_out_change"])


def test_no_foresight_models_never_see_target_day_weather():
    leaky = {"t_out_mean_target", "t_out_min_target", "rh_out_mean_target", "t_out_change", "physics_persistence"}
    assert not leaky & set(FEATURES["lightgbm_no_foresight"])
    assert not leaky & set(FEATURES["lightgbm_forecast"]) - {"physics_persistence"}
    assert "physics_persistence" not in FEATURES["hybrid_forecast"]


def test_subsets_pick_low_income_and_pre_1919_homes():
    from mould_risk.dataset import BUILD_ERA_ORDER
    from mould_risk.evaluate import PRE_1919, subsets
    assert BUILD_ERA_ORDER[PRE_1919] == "1900-1918"
    df = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-05", "2017-07-05", "2017-01-06"]),
        "equivalised_income": ["below_median", "above_median", "missing"],
        "build_era_ord": [1, 5, 2],
    })
    sizes = {name: len(part) for name, part in subsets(df)}
    assert sizes == {"all": 3, "heating season": 2, "below median income": 1, "pre-1919": 2}


def test_colder_walls_never_mean_fewer_risky_hours():
    room = synthetic_room()
    shares = [room_rows(room, f_rsi=f)["target"].mean() for f in (0.6, 0.7, 0.75, 0.85)]
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    assert shares[0] > shares[-1]


def synthetic_homes(n_homes=5):
    from mould_risk.dataset import ROOM_FEATURES
    parts = []
    for i in range(n_homes):
        rows = room_rows(synthetic_room(cold_from=12 + 2 * i), location=f"loc{i % 2}")
        rows = rows.assign(roomid=i, homeid=i, equivalised_income="below_median", **{c: 1 for c in ROOM_FEATURES})
        parts.append(rows)
    data = pd.concat(parts, ignore_index=True)
    for c in ("type", "clothesdrying", "hometype"):
        data[c] = data[c].astype("category")
    return data


def test_sweep_scores_only_the_requested_models():
    from mould_risk.evaluate import BASELINES, cross_validate
    from mould_risk.sensitivity import score
    df, importance = cross_validate(synthetic_homes(), 1, ["lightgbm_no_foresight"])
    assert importance is None and "lightgbm_forecast" not in df
    assert df["lightgbm_no_foresight"].between(0, 24).all()
    rows = pd.DataFrame(score(df, 0.7, 1, ["lightgbm_no_foresight"]))
    assert set(rows["model"]) == set(BASELINES) | {"lightgbm_no_foresight"}
    assert (rows["f_rsi"] == 0.7).all() and rows["high_risk_share"].between(0, 1).all()
