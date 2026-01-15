import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from joblib import load


def read_excel_safely(path, sheet_arg):
    """Same style as your other scripts: allow sheet index or name."""
    if sheet_arg is not None:
        try:
            return pd.read_excel(path, sheet_name=int(sheet_arg))
        except (ValueError, TypeError):
            pass
    try:
        return pd.read_excel(path, sheet_name=sheet_arg if sheet_arg is not None else 0)
    except Exception:
        xls = pd.ExcelFile(path)
        print(f"Requested sheet '{sheet_arg}' not found. Available sheets: {xls.sheet_names}. Falling back to first.")
        return pd.read_excel(path, sheet_name=0)


def main():
    ap = argparse.ArgumentParser(description="SHAP-based analysis for XGBoost pMOS model")
    ap.add_argument("--model", required=True, help="Path to pMOS_xgb.joblib (XGBoost pipeline)")
    ap.add_argument("--excel", required=True, help="Path to all_8448_with_pMOS.xlsx")
    ap.add_argument("--sheet", default=0, help="Sheet index or name (default: 0)")
    ap.add_argument("--out_dir", default="xgb_results", help="Directory to save SHAP figures")
    ap.add_argument("--max_samples", type=int, default=2000,
                    help="Optional subsampling of rows for faster SHAP (default: 2000)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading model pipeline from: {args.model}")
    pipe = load(args.model)
    pre = pipe.named_steps["pre"]
    xgb = pipe.named_steps["model"]

    print(f"[INFO] Loading Excel data from: {args.excel}")
    df = read_excel_safely(args.excel, args.sheet)
    df.columns = [c.strip().lower() for c in df.columns]

    # Columns we used for training
    needed = {"video_name", "character", "view", "compression", "qp", "pmos"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}. Got: {list(df.columns)}")

    # Ensure numeric types
    for c in ["qp", "vmaf", "ssim", "psnr", "pmos"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Engineered feature
    df["qp2"] = df["qp"] ** 2

    cat_cols = ["character", "view", "compression"]
    num_cols = ["qp", "qp2"] + [c for c in ["vmaf", "ssim", "psnr"] if c in df.columns]
    feat_cols = cat_cols + num_cols

    # Drop rows with missing feature values
    before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with NaN in features; kept {after}.")

    X_raw = df[feat_cols].copy()

    print("[INFO] Transforming features with the pipeline preprocessor…")
    X_trans = pre.transform(X_raw)

    # Optional subsampling for speed
    if args.max_samples is not None and after > args.max_samples:
        idx = np.random.RandomState(42).choice(after, size=args.max_samples, replace=False)
        X_trans_sample = X_trans[idx]
        print(f"[INFO] Subsampled from {after} to {len(idx)} samples for SHAP.")
    else:
        X_trans_sample = X_trans
        print(f"[INFO] Using all {after} samples for SHAP.")

    print("[INFO] Building SHAP TreeExplainer (XGBoost)…")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_trans_sample)

    # Feature names: from preprocessor if available, else generic
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_trans_sample.shape[1])]

    # 1) SHAP bar plot (global importance)
    print("[INFO] Creating SHAP bar plot…")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_trans_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    out_bar = os.path.join(args.out_dir, "xgb_shap_importance_bar.png")
    plt.tight_layout()
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print(f"[OK] Saved SHAP bar plot -> {out_bar}")

    # 2) SHAP beeswarm plot (optional, more detailed)
    print("[INFO] Creating SHAP beeswarm plot…")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_trans_sample,
        feature_names=feature_names,
        show=False
    )
    out_bee = os.path.join(args.out_dir, "xgb_shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(out_bee, dpi=300)
    plt.close()
    print(f"[OK] Saved SHAP beeswarm plot -> {out_bee}")

    print("[DONE] SHAP analysis complete.")


if __name__ == "__main__":
    main()
