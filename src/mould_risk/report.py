"""Frame results against persistence, the baseline that is hardest to beat 1-3 days ahead.

Persistence (tomorrow = today) scores well because high-risk days come in runs, but it cannot warn
of a run starting. So the report leads with skill over persistence, with home-level bootstrap
intervals, and with how many days ahead each high-risk episode was flagged.
"""

import numpy as np
import pandas as pd

from mould_risk.dataset import HIGH_RISK_HOURS, HORIZONS
from mould_risk.evaluate import BASELINES, FEATURES, HEADLINE
from mould_risk.paths import PROCESSED

BASE = "persistence"
BOOTSTRAP = 1000


def skill_table(results):
    """MAE skill (1 - MAE/MAE_persistence) and F1 / recall gain over persistence, per horizon and subset."""
    base = results.loc[results["model"] == BASE, ["horizon", "subset", "mae", "f1", "recall"]]
    out = results.merge(base, on=["horizon", "subset"], suffixes=("", "_base"))
    out["mae_skill"] = 1 - out["mae"] / out["mae_base"]
    out["f1_gain"] = out["f1"] - out["f1_base"]
    out["recall_gain"] = out["recall"] - out["recall_base"]
    return out[["horizon", "subset", "model", "n", "mae", "mae_skill", "f1", "f1_gain", "recall", "recall_gain",
                "onset_recall", "onset_n"]]


def home_stats(df, model):
    """Per-home sums so the bootstrap can resample homes without touching rows."""
    p, b, t = (df[c].to_numpy() >= HIGH_RISK_HOURS for c in (model, BASE, "target"))
    parts = pd.DataFrame({
        "homeid": df["homeid"].to_numpy(),
        "ae_m": np.abs(df[model] - df["target"]).to_numpy(), "ae_b": np.abs(df[BASE] - df["target"]).to_numpy(),
        "tp_m": p & t, "pp_m": p, "tp_b": b & t, "pp_b": b, "pos": t,
    })
    return parts.groupby("homeid").sum()


def f1(tp, pp, pos):
    with np.errstate(all="ignore"):
        return np.where(pp + pos > 0, 2 * tp / (pp + pos), np.nan)


def bootstrap_gain(df, model, n=BOOTSTRAP, seed=0):
    """Point estimate and 95 % interval for MAE skill and F1 gain, resampling whole homes."""
    s = home_stats(df, model)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=n)  # (n, homes) resample counts
    tot = w @ s.to_numpy(float)
    c = {k: tot[:, i] for i, k in enumerate(s.columns)}
    skill = 1 - c["ae_m"] / c["ae_b"]
    gain = f1(c["tp_m"], c["pp_m"], c["pos"]) - f1(c["tp_b"], c["pp_b"], c["pos"])
    a = s.sum()
    return {
        "model": model, "homes": len(s),
        "mae_skill": float(1 - a["ae_m"] / a["ae_b"]), "mae_skill_lo": float(np.nanpercentile(skill, 2.5)),
        "mae_skill_hi": float(np.nanpercentile(skill, 97.5)),
        "f1_gain": float(f1(a["tp_m"], a["pp_m"], a["pos"]) - f1(a["tp_b"], a["pp_b"], a["pos"])),
        "f1_gain_lo": float(np.nanpercentile(gain, 2.5)), "f1_gain_hi": float(np.nanpercentile(gain, 97.5)),
    }


def episode_starts(preds):
    """(roomid, start day) for each run of high-risk days, from the horizon-1 rows.

    At horizon 1, `persistence` is the day before the target day, so a start is target high, persistence low.
    """
    h1 = preds[preds["horizon"] == 1]
    start = h1[(h1["target"] >= HIGH_RISK_HOURS) & (h1["persistence"] < HIGH_RISK_HOURS)]
    return pd.DataFrame({"roomid": start["roomid"].to_numpy(), "start": (start["date"] + pd.Timedelta(days=1)).to_numpy()})


def lead_times(preds, model):
    """Share of high-risk episodes flagged at least h days before their first day.

    "At least 2 days" means the forecasts issued 1 and 2 days before the start both flagged it.
    """
    starts = episode_starts(preds)
    warned = []
    for h in HORIZONS:
        rows = preds[preds["horizon"] == h]
        issued = rows.assign(start=rows["date"] + pd.Timedelta(days=h))[["roomid", "start", model]]
        hit = starts.merge(issued, on=["roomid", "start"], how="left")[model] >= HIGH_RISK_HOURS
        warned.append(hit.to_numpy())
    cum = np.cumprod(np.vstack(warned), axis=0).astype(bool)
    return {"model": model, "episodes": len(starts), **{f"lead_{h}d": float(cum[i].mean()) if len(starts) else np.nan
                                                       for i, h in enumerate(HORIZONS)}}


def main():
    results = pd.read_csv(PROCESSED / "results.csv")
    preds = pd.read_parquet(PROCESSED / "predictions.parquet")
    skill = skill_table(results)
    skill.to_csv(PROCESSED / "skill_vs_persistence.csv", index=False)
    models = [m for m in BASELINES + list(FEATURES) if m != BASE]
    boot = pd.DataFrame([{"horizon": h, **bootstrap_gain(preds[preds["horizon"] == h], m)}
                         for h in HORIZONS for m in models])
    boot.to_csv(PROCESSED / "bootstrap_vs_persistence.csv", index=False)
    lead = pd.DataFrame([lead_times(preds, m) for m in BASELINES + list(FEATURES)])
    lead.to_csv(PROCESSED / "lead_times.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(skill[skill["subset"] == "all"].drop(columns="subset"))
        print(boot)
        print(lead)
    head = boot[boot["model"] == HEADLINE].set_index("horizon")
    for h, r in head.iterrows():
        print(f"h{h} {HEADLINE}: MAE skill {r.mae_skill:+.1%} [{r.mae_skill_lo:+.1%}, {r.mae_skill_hi:+.1%}], "
              f"F1 gain {r.f1_gain:+.3f} [{r.f1_gain_lo:+.3f}, {r.f1_gain_hi:+.3f}]")


if __name__ == "__main__":
    main()
