import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mould_risk.serve import Forecaster, band, create_app, fit
from test_pipeline import synthetic_homes


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = synthetic_homes()
    model_dir = tmp_path_factory.mktemp("models")
    fit(data, out=model_dir)
    return TestClient(create_app(Forecaster(data, model_dir)))


def test_band_thresholds():
    assert [band(h) for h in (0, 1.9, 2, 5.9, 6, 24)] == ["low", "low", "moderate", "moderate", "high", "high"]


def test_health_rooms_and_dashboard(client):
    assert client.get("/health").json()["rooms"] == 5
    rooms = client.get("/rooms").json()
    assert {r["homeid"] for r in rooms} == set(range(5))
    page = client.get("/")
    assert page.status_code == 200 and "Mould risk forecast" in page.text


def test_room_forecast_gives_three_horizons_within_range(client):
    out = client.get("/rooms/0/forecast", params={"date": "2017-01-10"}).json()
    f = out["forecasts"]
    assert [x["horizon"] for x in f] == [1, 2, 3]
    assert [x["target_date"] for x in f] == ["2017-01-11", "2017-01-12", "2017-01-13"]
    assert all(0 <= x["hours"] <= 24 and x["band"] in ("low", "moderate", "high") for x in f)


def test_history_ends_on_requested_day(client):
    hist = client.get("/rooms/0/history", params={"end": "2017-01-20", "days": 5}).json()
    assert [d["date"] for d in hist] == [str(d.date()) for d in pd.date_range("2017-01-16", "2017-01-20")]


def test_home_forecast_and_missing_ids(client):
    home = client.get("/homes/2/forecast", params={"date": "2017-01-10"}).json()
    assert len(home) == 1 and len(home[0]["forecasts"]) == 3
    assert client.get("/rooms/999/forecast", params={"date": "2017-01-10"}).status_code == 404
    assert client.get("/homes/999/forecast", params={"date": "2017-01-10"}).status_code == 404
    assert client.get("/rooms/0/forecast", params={"date": "2030-01-01"}).status_code == 404
