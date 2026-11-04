"""Home-grouped cross-validation of baselines vs LightGBM, feature attributions, and the survey check."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from mould_risk.dataset import HIGH_RISK_HOURS, HORIZONS, ROOM_FEATURES
from mould_risk.paths import PROCESSED, SURVEYS

WEATHER_AND_STATE = [
    "fav_lag1", "fav_lag2", "fav_mean7", "vtt_surface", "t_in_mean", "t_in_min", "rh_in_mean", "rh_in_max",
    "vp_excess", "t_out_mean", "t_out_min", "rh_out_mean", "t_out_mean_target", "t_out_min_target",
    "rh_out_mean_target", "t_out_change", "doy_sin", "doy_cos", "weekend_target", "persistence",
]
FEATURES = {
    "lightgbm": WEATHER_AND_STATE + ROOM_FEATURES,
    "hybrid": WEATHER_AND_STATE + ROOM_FEATURES + ["physics_persistence"],
}
PARAMS = {"objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 7}
ROUNDS = 400
FOLDS = 5


def classify(pred, truth):
    p, t = pred >= HIGH_RISK_HOURS, truth >= HIGH_RISK_HOURS
    tp = (p & t).sum()
    precision = tp / p.sum() if p.sum() else np.nan
    recall = tp / t.sum() if t.sum() else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    return precision, recall, f1


def metrics(df, col):
    out = {"mae": float(np.mean(np.abs(df[col] - df["target"])))}
    out["precision"], out["recall"], out["f1"] = map(float, classify(df[col].to_numpy(), df["target"].to_numpy()))
    onset = df[(df["persistence"] < HIGH_RISK_HOURS) & (df["target"] >= HIGH_RISK_HOURS)]
    out["onset_recall"] = float((onset[col] >= HIGH_RISK_HOURS).mean()) if len(onset) else np.nan
    out["onset_n"] = int(len(onset))
    return out


def cross_validate(data, h):
    df = data[data["horizon"] == h].reset_index(drop=True)
    folds = GroupKFold(n_splits=FOLDS).split(df, groups=df["homeid"])
    contrib = {}
    for train, test in folds:
        for name, cols in FEATURES.items():
            model = lgb.train(PARAMS, lgb.Dataset(df.loc[train, cols], df.loc[train, "target"]), ROUNDS)
            df.loc[test, name] = model.predict(df.loc[test, cols])
            if name == "lightgbm":
                c = model.predict(df.loc[test, cols], pred_contrib=True)[:, :-1]
                contrib.setdefault(name, []).append(pd.DataFrame(np.abs(c), columns=cols))
    df[["lightgbm", "hybrid"]] = df[["lightgbm", "hybrid"]].clip(0, 24)
    importance = pd.concat(contrib["lightgbm"]).mean().sort_values(ascending=False)
    return df, importance


def results_table(preds):
    rows = []
    for h, df in preds.items():
        for subset, part in (("all", df), ("heating season", df[df["date"].dt.month.isin([10, 11, 12, 1, 2, 3, 4])])):
            for model in ("persistence", "physics_persistence", "lightgbm", "hybrid"):
                rows.append({"horizon": h, "subset": subset, "model": model, "n": len(part), **metrics(part, model)})
    return pd.DataFrame(rows)


def survey_check(data):
    """Do homes with more high-risk room-days report damp more often?"""
    survey = pd.read_csv(SURVEYS / "survey_responses.csv", low_memory=False)
    score = pd.to_numeric(survey["damp_probs_1d_end"].astype(str).str.extract(r"^(\d)")[0], errors="coerce")
    damp = pd.DataFrame({"homeid": survey["homeid"], "damp_score": score}).dropna().groupby("homeid").mean()
    h1 = data[data["horizon"] == 1]
    risk = h1.groupby("homeid").agg(
        high_risk_share=("target", lambda s: float((s >= HIGH_RISK_HOURS).mean())),
        vtt_max=("vtt_surface", "max"),
    )
    joined = risk.join(damp, how="inner")
    rho, p = stats.spearmanr(joined["high_risk_share"], joined["damp_score"])
    rho_v, p_v = stats.spearmanr(joined["vtt_max"], joined["damp_score"])
    agree = joined["damp_score"] >= 5
    u = stats.mannwhitneyu(joined.loc[agree, "high_risk_share"], joined.loc[~agree, "high_risk_share"],
                           alternative="greater") if agree.any() and (~agree).any() else None
    return {
        "homes": int(len(joined)),
        "homes_reporting_damp": int(agree.sum()),
        "spearman_high_risk_share": [float(rho), float(p)],
        "spearman_vtt_max": [float(rho_v), float(p_v)],
        "mannwhitney_p_reporting_vs_not": float(u.pvalue) if u else None,
        "median_share_reporting": float(joined.loc[agree, "high_risk_share"].median()) if agree.any() else None,
        "median_share_not": float(joined.loc[~agree, "high_risk_share"].median()),
    }, joined


def main():
    data = pd.read_parquet(PROCESSED / "room_days.parquet")
    preds, importances = {}, {}
    for h in HORIZONS:
        preds[h], importances[h] = cross_validate(data, h)
        print(f"horizon {h} done", flush=True)
    table = results_table(preds)
    table.to_csv(PROCESSED / "results.csv", index=False)
    pd.DataFrame(importances).to_csv(PROCESSED / "importance.csv")
    pd.concat(preds.values()).to_parquet(PROCESSED / "predictions.parquet", index=False)
    survey, joined = survey_check(data)
    joined.to_csv(PROCESSED / "survey_check.csv")
    (PROCESSED / "survey_check.json").write_text(json.dumps(survey, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table)
        print(pd.DataFrame(importances).head(12))
    print(json.dumps(survey, indent=2))


if __name__ == "__main__":
    main()
