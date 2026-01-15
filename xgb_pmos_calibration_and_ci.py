import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

def detect_columns(df):
    cols_lower = {c.lower(): c for c in df.columns}

    true_candidates = ["pmos_true", "y_true", "mos_true", "target"]
    pred_candidates = ["pmos_pred", "y_pred", "mos_pred", "pred"]

    y_true_col = None
    y_pred_col = None

    for name in true_candidates:
        if name in cols_lower:
            y_true_col = cols_lower[name]
            break

    for name in pred_candidates:
        if name in cols_lower:
            y_pred_col = cols_lower[name]
            break

    if y_true_col is None or y_pred_col is None:
        raise ValueError(
            f"Could not detect true/pred columns. Found: {list(df.columns)}. "
            f"Expected e.g. pmos_true & pmos_pred."
        )

    return y_true_col, y_pred_col

def metrics_all(y, y_hat):
    r2 = r2_score(y, y_hat)
    # For older sklearn versions without 'squared' parameter:
    mse = mean_squared_error(y, y_hat)
    rmse = mse ** 0.5
    pear = pearsonr(y, y_hat)[0] if len(y) > 1 else np.nan
    spear = spearmanr(y, y_hat)[0] if len(y) > 1 else np.nan
    return r2, rmse, pear, spear


def bootstrap_ci(y, y_hat, n_boot=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y)

    r2_list = []
    plcc_list = []
    srcc_list = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        y_hat_b = y_hat[idx]

        # R2
        try:
            r2_b = r2_score(y_b, y_hat_b)
        except Exception:
            r2_b = np.nan

        # Pearson
        try:
            plcc_b = pearsonr(y_b, y_hat_b)[0]
        except Exception:
            plcc_b = np.nan

        # Spearman
        try:
            srcc_b = spearmanr(y_b, y_hat_b)[0]
        except Exception:
            srcc_b = np.nan

        r2_list.append(r2_b)
        plcc_list.append(plcc_b)
        srcc_list.append(srcc_b)

    def ci(a):
        a = np.array(a, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return (np.nan, np.nan)
        low = float(np.percentile(a, 2.5))
        high = float(np.percentile(a, 97.5))
        return (low, high)

    return {
        "R2_95CI": ci(r2_list),
        "PLCC_95CI": ci(plcc_list),
        "SRCC_95CI": ci(srcc_list),
    }

def main():
    ap = argparse.ArgumentParser(
        description="Linear calibration + bootstrap 95% CI for XGBoost pMOS OOF predictions."
    )
    ap.add_argument(
        "--oof_csv",
        required=True,
        help="Path to OOF CSV, e.g. xgb_results/oof_pmos_8448.csv"
    )
    ap.add_argument(
        "--out_json",
        default="xgb_results/xgb_calibration_and_ci.json",
        help="Where to save calibration + CI summary JSON"
    )
    ap.add_argument(
        "--n_boot",
        type=int,
        default=2000,
        help="Number of bootstrap samples (default 2000)"
    )
    args = ap.parse_args()

    print(f"[INFO] Loading OOF predictions from: {args.oof_csv}")
    df = pd.read_csv(args.oof_csv)

    y_true_col, y_pred_col = detect_columns(df)
    print(f"[INFO] Detected true column: {y_true_col}")
    print(f"[INFO] Detected pred column: {y_pred_col}")

    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values
    print(f"[INFO] Samples: {len(y_true)}")

    # 1) Raw metrics
    r2_raw, rmse_raw, plcc_raw, srcc_raw = metrics_all(y_true, y_pred)
    print(f"[BEFORE calibration] R2={r2_raw:.4f} | RMSE={rmse_raw:.4f} | "
          f"PLCC={plcc_raw:.4f} | SRCC={srcc_raw:.4f}")

    # 2) Linear calibration: y_cal = a * y_pred + b
    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_true)
    a = float(lr.coef_[0])
    b = float(lr.intercept_)

    y_cal = lr.predict(y_pred.reshape(-1, 1))

    r2_cal, rmse_cal, plcc_cal, srcc_cal = metrics_all(y_true, y_cal)
    print(f"[AFTER calibration]  R2={r2_cal:.4f} | RMSE={rmse_cal:.4f} | "
          f"PLCC={plcc_cal:.4f} | SRCC={srcc_cal:.4f}")
    print(f"[CALIBRATION] y_cal = {a:.6f} * y_pred + {b:.6f}")

    # 3) Bootstrap CIs on calibrated predictions
    print(f"[INFO] Bootstrapping {args.n_boot} samples for 95%% CIs (calibrated)...")
    cis_cal = bootstrap_ci(y_true, y_cal, n_boot=args.n_boot, random_state=42)

    print(
        f"[CI] R2 95% CI:   {cis_cal['R2_95CI'][0]:.4f} .. {cis_cal['R2_95CI'][1]:.4f}\n"
        f"[CI] PLCC 95% CI: {cis_cal['PLCC_95CI'][0]:.4f} .. {cis_cal['PLCC_95CI'][1]:.4f}\n"
        f"[CI] SRCC 95% CI: {cis_cal['SRCC_95CI'][0]:.4f} .. {cis_cal['SRCC_95CI'][1]:.4f}"
    )

    # 4) Store calibrated predictions back into CSV
    df["pmos_calibrated"] = y_cal
    df.to_csv(args.oof_csv, index=False)
    print(f"[INFO] Updated OOF CSV with 'pmos_calibrated' -> {args.oof_csv}")

    # 5) Save JSON summary
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        "note": "Model trained on pMOS (1–5). Linear calibration applied on OOF predictions.",
        "calibration": {
            "a": a,
            "b": b,
            "metrics_before": {
                "R2": r2_raw,
                "RMSE": rmse_raw,
                "PLCC": plcc_raw,
                "SRCC": srcc_raw,
            },
            "metrics_after": {
                "R2": r2_cal,
                "RMSE": rmse_cal,
                "PLCC": plcc_cal,
                "SRCC": srcc_cal,
            },
        },
        "bootstrap_95CI_calibrated": cis_cal,
        "n_boot": args.n_boot,
    }

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Saved calibration + CI summary -> {args.out_json}")
    print("[DONE] Calibration and bootstrap CI complete.")

if __name__ == "__main__":
    main()
