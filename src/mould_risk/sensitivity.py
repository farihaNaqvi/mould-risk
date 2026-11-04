"""fRsi sensitivity sweep.

The target (hours with surface RH >= 80 %) depends on one assumed wall factor, fRsi. 0.75 is the
BS 5250 / ISO 13788 design value; cold corners and solid pre-1919 walls can sit nearer 0.6. This
rebuilds the table across a range and checks whether the prevalence, the model ranking and the
survey link hold up.
"""

import pandas as pd

from mould_risk.dataset import HIGH_RISK_HOURS, HORIZONS, build, load_hourly, room_table
from mould_risk.evaluate import BASELINES, cross_validate, metrics, subsets, survey_check
from mould_risk.paths import PROCESSED

F_RSI_GRID = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
SWEEP_MODELS = ["lightgbm_forecast", "hybrid_forecast", "lightgbm_no_foresight"]


def score(df, f_rsi, h, models):
    rows = []
    for subset, part in subsets(df):
        share = float((part["target"] >= HIGH_RISK_HOURS).mean())
        for m in BASELINES + list(models):
            rows.append({"f_rsi": f_rsi, "horizon": h, "subset": subset, "model": m, "n": len(part),
                         "high_risk_share": share, **metrics(part, m)})
    return rows


def sweep(grid=F_RSI_GRID, horizons=HORIZONS, models=SWEEP_MODELS):
    rooms = room_table()
    hourly = load_hourly(rooms)
    rows, survey = [], []
    for f in grid:
        data = build(f_rsi=f, rooms=rooms, hourly=hourly)
        for h in horizons:
            df, _ = cross_validate(data, h, models)
            rows += score(df, f, h, models)
        s, _ = survey_check(data)
        survey.append({"f_rsi": f, "homes": s["homes"],
                       "rho_high_risk_share": s["spearman_high_risk_share"][0],
                       "p_high_risk_share": s["spearman_high_risk_share"][1],
                       "mannwhitney_p": s["mannwhitney_p_reporting_vs_not"]})
        print(f"f_rsi {f:.2f} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(survey)


def main():
    table, survey = sweep()
    table.to_csv(PROCESSED / "sensitivity_frsi.csv", index=False)
    survey.to_csv(PROCESSED / "sensitivity_frsi_survey.csv", index=False)
    view = table[table["subset"] == "all"].pivot_table(
        index=["horizon", "f_rsi"], columns="model", values="f1").round(3)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[table["subset"] == "all"].groupby("f_rsi")["high_risk_share"].first().round(3))
        print(view)
        print(survey.round(3))


if __name__ == "__main__":
    main()
