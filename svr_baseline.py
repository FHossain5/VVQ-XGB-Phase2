import argparse, math, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

def eval_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    spea = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    return rmse, mae, r2, pear, spea

def aggregate_duplicates(df, key):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = {c:"mean" for c in num}
    for c in df.columns:
        if c not in num: agg[c] = "first"
    return df.groupby(key, as_index=False).agg(agg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--group_by", default="video_name")
    ap.add_argument("--out_prefix", default="svr")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]
    comps = ["clarity","depth","interaction","overall"]
    if all(c in df.columns for c in comps):
        df["mos"] = df[comps].astype(float).mean(axis=1).clip(1,5)

    for c in ["vmaf","psnr","ssim","mos"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    df = aggregate_duplicates(df, args.group_by)

    feat_cols = [c for c in ["vmaf","psnr","ssim"] if c in df.columns]
    df = df.dropna(subset=feat_cols + ["mos", args.group_by])

    X = df[feat_cols].values
    y = df["mos"].values
    groups = df[args.group_by].astype(str).values

    # Standardize + SVR(RBF)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=10.0, epsilon=0.1, gamma="scale", kernel="rbf"))
    ])

    # OOF
    gkf = GroupKFold(n_splits=args.cv_splits)
    from sklearn.base import clone
    y_oof = np.zeros_like(y, dtype=float)
    for i,(tr,va) in enumerate(gkf.split(X, y, groups=groups), 1):
        p = clone(pipe)
        p.fit(X[tr], y[tr])
        y_hat = p.predict(X[va])
        rmse, mae, r2, pear, spea = eval_metrics(y[va], y_hat)
        print(f"[Fold {i}] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Pearson={pear:.4f} | Spearman={spea:.4f}")
        y_oof[va] = y_hat
    rmse, mae, r2, pear, spea = eval_metrics(y, y_oof)
    print(f"[OOF overall] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Pearson={pear:.4f} | Spearman={spea:.4f}")

if __name__ == "__main__":
    main()


