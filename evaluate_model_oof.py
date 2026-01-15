import argparse, math
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

def eval_metrics(y_true, y_pred, tag="OOF"):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    spea = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    print(f"[{tag}] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Pearson={pear:.4f} | Spearman={spea:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pear": pear, "spea": spea}

def aggregate_duplicates(df, group_key):
    # average numeric cols, take first for categoricals
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = {c: "mean" for c in num}
    for c in df.columns:
        if c not in num: agg[c] = "first"
    return df.groupby(group_key, as_index=False).agg(agg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv",   required=True)
    ap.add_argument("--target", default="mos", choices=["mos","dmos_100"])
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--group_by", default="video_name")
    ap.add_argument("--aggregate_by", default="video_name", help="Average duplicate ratings per this key (use video_name)")
    args = ap.parse_args()

    # Load pipeline + its expected features
    bundle = load(args.model)
    pipe   = bundle["pipeline"]
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]

    # Load labeled data
    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # Recompute MOS from components (safer)
    comps = ["clarity","depth","interaction","overall"]
    if all(c in df.columns for c in comps):
        df["mos"] = df[comps].astype(float).mean(axis=1)

    # MOS_ref & DMOS_100 if needed
    if "mos" in df.columns and "mos_ref" not in df.columns:
        df["mos_ref"] = df.groupby(["character","view"])["mos"].transform("max")
    if "dmos_100" not in df.columns and {"mos_ref","mos"}.issubset(df.columns):
        df["dmos_100"] = 25.0 * (df["mos_ref"] - df["mos"])

    # Ensure numerics
    for c in ["vmaf","ssim","psnr","qp"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Create qp2 if expected
    if "qp2" in num_cols and "qp2" not in df.columns:
        df["qp2"] = (pd.to_numeric(df["qp"], errors="coerce") ** 2)

    # Aggregate duplicates like in training
    if args.aggregate_by in df.columns:
        df = aggregate_duplicates(df, args.aggregate_by)

    # Features
    feats = [c for c in (cat_cols + num_cols) if c in df.columns]
    df = df.dropna(subset=feats + [args.target, args.group_by])

    y = df[args.target].astype(float).values
    X = df[feats]
    groups = df[args.group_by].astype(str).values

    # GroupKFold OOF
    gkf = GroupKFold(n_splits=args.cv_splits)
    y_oof = np.zeros_like(y, dtype=float)
    for i,(tr,va) in enumerate(gkf.split(X, y, groups=groups), 1):
        # fit on train split
        pipe.fit(X.iloc[tr], y[tr])
        y_oof[va] = pipe.predict(X.iloc[va])
        print(f"fold {i}: n_train={len(tr)} n_valid={len(va)}")

    eval_metrics(y, y_oof, tag="OOF (aggregated)")
    out = df[[args.group_by]].copy()
    out["y_true"] = y; out["y_pred"] = y_oof
    out.to_csv(f"oof_{args.target}_aggregated.csv", index=False)
    print(f"Saved OOF preds -> oof_{args.target}_aggregated.csv")

if __name__ == "__main__":
    main()
