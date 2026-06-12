from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "data" / "processed" / "analysis_panel.csv"
MAIN_TABLE = PROJECT_ROOT / "output" / "tables" / "table_regression_specifications.csv"
OUT = PROJECT_ROOT / "output" / "tables" / "referee2_core_replication.csv"

EXPOSURE = "observed_exposure_cosine_nearest"
EVENT_MONTH_ID = 2022 * 12 + 9


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ coef


def cluster_se_one_regressor(x: np.ndarray, u: np.ndarray, clusters: pd.Series, k_params: int) -> tuple[float, float]:
    x = x.reshape(-1, 1)
    xtx = float((x.T @ x).item())
    meat = 0.0
    cluster_codes = pd.Series(clusters.to_numpy(), index=np.arange(len(clusters)))
    for _, idx in cluster_codes.groupby(cluster_codes).groups.items():
        pos = list(idx)
        score_g = float((x[pos].T @ u[pos]).item())
        meat += score_g * score_g
    n = len(u)
    g = clusters.nunique()
    correction = (g / (g - 1)) * ((n - 1) / (n - k_params))
    var = correction * meat / (xtx * xtx)
    var_uncorrected = meat / (xtx * xtx)
    return float(np.sqrt(var)), float(np.sqrt(var_uncorrected))


def main() -> None:
    df = pd.read_csv(PANEL, dtype={"cno4": "string"})
    threshold = float(df[EXPOSURE].quantile(0.75))
    df = df.loc[(df[EXPOSURE] > threshold) | (df[EXPOSURE] == 0)].copy()
    df["treat"] = df[EXPOSURE] > threshold
    df["post"] = df["month_id"] > EVENT_MONTH_ID
    df["did_post"] = df["treat"].astype(int) * df["post"].astype(int)
    df = df.dropna(subset=["ln_parados", "did_post", "cno4", "period"]).copy().reset_index(drop=True)

    y = df["ln_parados"].to_numpy(dtype=float)
    d = df["did_post"].to_numpy(dtype=float)

    fe = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="const"),
            pd.get_dummies(df["cno4"], prefix="cno4", drop_first=True, dtype=float),
            pd.get_dummies(df["period"], prefix="period", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    z = fe.to_numpy(dtype=float)
    y_res = residualize(y, z)
    d_res = residualize(d, z)
    beta = float((d_res @ y_res) / (d_res @ d_res))
    u = y_res - beta * d_res
    se, se_uncorrected = cluster_se_one_regressor(d_res, u, df["cno4"], k_params=z.shape[1] + 1)

    main = pd.read_csv(MAIN_TABLE)
    target = main.loc[main["spec"].eq("1. Main DiD")].iloc[0]
    row = {
        "audit": "referee2_core_fwl",
        "nobs": int(len(df)),
        "clusters": int(df["cno4"].nunique()),
        "threshold_p75": threshold,
        "beta_fwl": beta,
        "beta_main_table": float(target["beta_log_points"]),
        "beta_abs_diff": abs(beta - float(target["beta_log_points"])),
        "se_cluster_fwl_corrected": se,
        "se_cluster_fwl_uncorrected": se_uncorrected,
        "se_main_table": float(target["se_cluster_cno4"]),
        "se_abs_diff_corrected": abs(se - float(target["se_cluster_cno4"])),
        "method_note": "FWL residualization with explicit FE dummies; cluster SE computed manually for one residualized regressor.",
    }
    pd.DataFrame([row]).to_csv(OUT, index=False)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
