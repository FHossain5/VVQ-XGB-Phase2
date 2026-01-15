# linear_baseline_vv_weightedscore.py
import argparse, math, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

def eval_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    spea = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    return dict(RMSE=rmse, MAE=mae, R2=r2, Pearson=pear, Spearman=spea)

def aggregate_duplicates(df, group_key):
    # Average numeric columns; take first categorical
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = {c: "mean" for c in num}
    for c in df.columns:
        if c not in num:
            agg[c] = "first"
    return df.groupby(group_key, as_index=False).agg(agg)

def main():
    ap = argparse.ArgumentParser(description="Linear baseline for VV-WeightedScore on rated subset")
    ap.add_argument("--csv", required=True, help="Path to rated CSV (e.g., vve_sessions_with_metrics.csv)")
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--group_by", default="video_name")
    ap.add_argument("--out_prefix", default="linear_baseline")
    args = ap.parse_args()

    # Load & normalize columns
    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # Recompute MOS from sub-scores if available, clip to [1,5]
    comps = ["clarity","depth","interaction","overall"]
    if all(c in df.columns for c in comps):
        df["mos"] = df[comps].astype(float).mean(axis=1).clip(1,5)

    # Create MOS_ref & DMOS_100 (useful if you also want to evaluate DMOS later)
    if "mos" in df.columns and "mos_ref" not in df.columns:
        df["mos_ref"] = df.groupby(["character","view"])["mos"].transform("max")
    if "dmos_100" not in df.columns and {"mos_ref","mos"}.issubset(df.columns):
        df["dmos_100"] = 25.0 * (df["mos_ref"] - df["mos"])

    # Ensure numeric types
    for c in ["vmaf","psnr","ssim","mos"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate duplicate ratings per video (same as training/eval protocol)
    if args.group_by in df.columns:
        df = aggregate_duplicates(df, args.group_by)

    # Select features (linear baseline on objective metrics)
    feat_cols = [c for c in ["vmaf","psnr","ssim"] if c in df.columns]
    if len(feat_cols) == 0:
        raise ValueError("No objective metric columns found. Need at least one of: VMAF, PSNR, SSIM.")
    target = "mos"
    needed = feat_cols + [target, args.group_by]
    df = df.dropna(subset=needed)

    X = df[feat_cols].values
    y = df[target].astype(float).values
    groups = df[args.group_by].astype(str).values

    # GroupKFold OOF CV
    gkf = GroupKFold(n_splits=args.cv_splits)
    y_oof = np.zeros_like(y, dtype=float)
    fold_rows = []

    for i, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        lr = LinearRegression()
        lr.fit(X[tr], y[tr])
        y_hat = lr.predict(X[va])
        m = eval_metrics(y[va], y_hat)
        m["fold"] = i
        fold_rows.append(m)
        print(f"[Fold {i}] RMSE={m['RMSE']:.4f} | MAE={m['MAE']:.4f} | R2={m['R2']:.4f} | Pearson={m['Pearson']:.4f} | Spearman={m['Spearman']:.4f}")
        y_oof[va] = y_hat

    overall = eval_metrics(y, y_oof)
    print(f"[OOF overall] RMSE={overall['RMSE']:.4f} | MAE={overall['MAE']:.4f} | R2={overall['R2']:.4f} | Pearson={overall['Pearson']:.4f} | Spearman={overall['Spearman']:.4f}")

    # Fit final linear model on ALL aggregated rated data to get VV-WeightedScore coefficients
    final_lr = LinearRegression()
    final_lr.fit(X, y)
    intercept = float(final_lr.intercept_)
    coefs = {feat_cols[i]: float(final_lr.coef_[i]) for i in range(len(feat_cols))}

    # Save coefficients (VV-WeightedScore) and metrics
    coef_df = pd.DataFrame([{"feature":"intercept","coef":intercept}] + [{"feature":k, "coef":v} for k,v in coefs.items()])
    coef_df.to_csv(f"{args.out_prefix}_vv_weights.csv", index=False)

    # Pretty formula string
    terms = " + ".join([f"{coefs[c]:.6f}*{c.upper()}" for c in feat_cols])
    formula = f"VV-WeightedScore = {intercept:.6f} + {terms}"
    with open(f"{args.out_prefix}_vv_formula.txt","w", encoding="utf-8") as f:
        f.write(formula+"\n")

    # Save OOF predictions and metrics
    oof_out = df[[args.group_by]].copy()
    oof_out["y_true"] = y
    oof_out["y_pred"] = y_oof
    oof_out.to_csv(f"{args.out_prefix}_oof_mos.csv", index=False)

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(f"{args.out_prefix}_cv_folds_metrics.csv", index=False)

    with open(f"{args.out_prefix}_cv_overall_metrics.json","w") as f:
        json.dump(overall, f, indent=2)

    # Also emit a one-row comparison CSV to append into your comparison table later
    comp = pd.DataFrame([{
        "Model": "Linear Regression (VV-WeightedScore)",
        "R2": overall["R2"],
        "RMSE": overall["RMSE"],
        "MAE": overall["MAE"],
        "Pearson": overall["Pearson"],
        "Spearman": overall["Spearman"]
    }])
    comp.to_csv(f"{args.out_prefix}_comparison_row.csv", index=False)

    # Console summary
    print("\n=== VV-WeightedScore (linear baseline) ===")
    print(formula)
    print("Coefficients saved to:", f"{args.out_prefix}_vv_weights.csv")
    print("OOF metrics (overall) saved to:", f"{args.out_prefix}_cv_overall_metrics.json")
    print("Fold metrics saved to:", f"{args.out_prefix}_cv_folds_metrics.csv")
    print("OOF predictions saved to:", f"{args.out_prefix}_oof_mos.csv")
    print("Comparison row saved to:", f"{args.out_prefix}_comparison_row.csv")

if __name__ == "__main__":
    main()
