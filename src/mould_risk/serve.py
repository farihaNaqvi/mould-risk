"""Thin serving layer: fit the headline model, then serve 1-3 day mould risk over HTTP with a dashboard.

IDEAL is historical, so the API replays it: give a room and an issue date and get the forecast the
model would have made that morning. The served model is fit on every home, so replayed scores are
in-sample; the honest accuracy numbers are the home-grouped cross-validation ones from `evaluate`.

    python -m mould_risk.serve fit     # train and save one model per horizon
    python -m mould_risk.serve run     # http://127.0.0.1:8000
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from mould_risk.dataset import HIGH_RISK_HOURS, HORIZONS
from mould_risk.evaluate import FEATURES, HEADLINE, PARAMS, ROUNDS
from mould_risk.paths import PROCESSED

MODEL_DIR = PROCESSED / "models"
CATEGORICAL = ("type", "clothesdrying", "hometype")
DASHBOARD = Path(__file__).with_name("dashboard.html")
MODERATE_HOURS = 2
ADVICE = {
    "high": "Surfaces likely to stay damp for 6+ hours. Heat the room steadily and ventilate after cooking, "
            "bathing or drying clothes; check cold corners and window reveals.",
    "moderate": "Some hours near the mould threshold. Keep trickle vents open and avoid drying clothes indoors.",
    "low": "No action needed.",
}


def band(hours):
    return "high" if hours >= HIGH_RISK_HOURS else "moderate" if hours >= MODERATE_HOURS else "low"


def fit(data, model=HEADLINE, out=MODEL_DIR):
    """Train one booster per horizon on all room-days and save it with its feature list."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cols = FEATURES[model]
    for h in HORIZONS:
        df = data[data["horizon"] == h]
        lgb.train(PARAMS, lgb.Dataset(df[cols], df["target"]), ROUNDS).save_model(out / f"{model}_h{h}.txt")
    (out / f"{model}.json").write_text(json.dumps({"model": model, "features": cols}))


class Forecaster:
    """Scores room-day rows with the saved boosters. Rows come from the `dataset` table."""

    def __init__(self, data, model_dir=MODEL_DIR, model=HEADLINE):
        meta = json.loads((Path(model_dir) / f"{model}.json").read_text())
        self.model, self.cols = model, meta["features"]
        self.boosters = {h: lgb.Booster(model_file=str(Path(model_dir) / f"{model}_h{h}.txt")) for h in HORIZONS}
        data = data.copy()
        for c in CATEGORICAL:  # the booster stores its training categories and remaps to them at predict
            if c in data and data[c].dtype != "category":
                data[c] = data[c].astype("category")
        self.data = data.set_index(["roomid", "date", "horizon"]).sort_index()

    def rooms(self):
        r = self.data.reset_index().drop_duplicates("roomid")
        out = r[["roomid", "homeid", "type"]].assign(type=lambda d: d["type"].astype(str))
        return out.sort_values(["homeid", "roomid"]).to_dict("records")

    def room_rows(self, roomid):
        if roomid not in self.data.index.get_level_values(0):
            raise KeyError(roomid)
        return self.data.loc[roomid]

    def dates(self, roomid):
        d = self.room_rows(roomid).index.get_level_values("date")
        return {"first": str(d.min().date()), "last": str(d.max().date())}

    def forecast(self, roomid, date):
        rows = self.room_rows(roomid)
        date = pd.Timestamp(date)
        out = []
        for h in HORIZONS:
            if (date, h) not in rows.index:
                continue
            row = rows.loc[[(date, h)]]
            hours = float(np.clip(self.boosters[h].predict(row[self.cols])[0], 0, 24))
            out.append({
                "horizon": h, "target_date": str((date + pd.Timedelta(days=h)).date()),
                "hours": round(hours, 1), "band": band(hours), "advice": ADVICE[band(hours)],
                "physics_hours": round(float(row["physics_forecast"].iloc[0]), 1),
                "observed_hours": round(float(row["target"].iloc[0]), 1),
            })
        return out

    def history(self, roomid, end=None, days=60):
        """Observed high-risk hours per day, read off the horizon-1 rows (target day = date + 1)."""
        h1 = self.room_rows(roomid).xs(1, level="horizon")["target"]
        h1.index = h1.index + pd.Timedelta(days=1)
        if end is not None:
            h1 = h1[h1.index <= pd.Timestamp(end)]
        h1 = h1.tail(days)
        return [{"date": str(d.date()), "hours": round(float(v), 1)} for d, v in h1.items()]

    def homes(self):
        return sorted({int(r["homeid"]) for r in self.rooms()})


def create_app(forecaster: Forecaster) -> FastAPI:
    app = FastAPI(title="mould-risk", description="1-3 day indoor mould risk forecasts (replay of IDEAL data)")

    def room_or_404(roomid):
        try:
            return forecaster.room_rows(roomid)
        except KeyError:
            raise HTTPException(404, f"room {roomid} not found")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        return DASHBOARD.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": forecaster.model, "rooms": len(forecaster.rooms())}

    @app.get("/rooms")
    def rooms():
        return forecaster.rooms()

    @app.get("/rooms/{roomid}/dates")
    def dates(roomid: int):
        room_or_404(roomid)
        return forecaster.dates(roomid)

    @app.get("/rooms/{roomid}/forecast")
    def forecast(roomid: int, date: str):
        room_or_404(roomid)
        out = forecaster.forecast(roomid, date)
        if not out:
            raise HTTPException(404, f"no data for room {roomid} on {date}")
        return {"roomid": roomid, "issued": date, "forecasts": out}

    @app.get("/rooms/{roomid}/history")
    def history(roomid: int, end: str | None = None, days: int = 60):
        room_or_404(roomid)
        return forecaster.history(roomid, end, days)

    @app.get("/homes/{homeid}/forecast")
    def home_forecast(homeid: int, date: str):
        rooms = [r for r in forecaster.rooms() if r["homeid"] == homeid]
        if not rooms:
            raise HTTPException(404, f"home {homeid} not found")
        return [{**r, "forecasts": forecaster.forecast(r["roomid"], date)} for r in rooms]

    return app


def default_app():
    return create_app(Forecaster(pd.read_parquet(PROCESSED / "room_days.parquet")))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fit":
        fit(pd.read_parquet(PROCESSED / "room_days.parquet"))
        print(f"saved to {MODEL_DIR}")
    else:
        import uvicorn
        uvicorn.run("mould_risk.serve:default_app", factory=True, host="127.0.0.1", port=8000)
