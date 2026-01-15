import argparse
import math
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import dump

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


def compute_mos_cols(df):
    # Recompute MOS from components if present; otherwise trust 'mos'
    comps = ["clarity", "depth", "interaction", "overall"]
    if all(c in df.columns for c in comps):
        df["mos_recalc"] = df[comps].astype(float).mean(axis=1)
        if "mos" not in df.columns:
            df["mos"] = df["mos_recalc"]
        else:
            # If 'mos' exists, prefer the explicit recompute to keep consistency
            df["mos"] = df["mos_recalc"]
    # Clamp to [1,5] if this was your rating scale
    if "mos" in df.columns:
        df["mos"] = df["mos"].astype(float).clip(lower=1.0, upper=5.0)
    return df


def compute_mosref_and_dmos(df):
    # MOS_ref = max MOS per (character, view)
    if not {"character", "view", "mos"}.issubset(df.columns):
        raise ValueError("Required columns missing to compute MOS_ref: character, view, mos")

    grp = df.groupby(["character", "view"])["mos"].transform("max")
    df["mos_ref"] = grp
    # DMOS_100 = 25 * (MOS_ref - MOS)
    df["dmos_100"] = 25.0 * (df["mos_ref"] - df["mos"])
    return df


def aggregate_duplicates(df, group_key):
    """
    Average repeated ratings per video (or per chosen key).
    Keeps categorical columns by first non-null; averages numeric.
    """
    if group_key not in df.columns:
        return df

    # Identify numeric columns to average
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep category columns by first()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    agg_map = {c: "first" for c in cat_cols}
    for c in numeric_cols:
        agg_map[c] = "mean"

    return df.groupby(group_key, as_index=False).agg(agg_map)


def eval_regression(y_true, y_pred, label="CV"):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    spea = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    print(f"[{label}] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Pearson={pear:.4f} | Spearman={spea:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pear": pear, "spea": spea}


def build_model(model_name="hgb", random_state=42):
    if model_name.lower() == "xgb" and HAVE_XGB:
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",
        )
        param_distributions = {
            "model__n_estimators": [300, 400, 500, 800],
            "model__learning_rate": [0.03, 0.05, 0.07],
            "model__max_depth": [4, 6, 8],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        }
        return model, param_distributions
    else:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=random_state
        )
        param_distributions = {
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_leaf_nodes": [31, 63, 127],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 0.5],
            "model__max_depth": [None, 8, 12],
        }
        return model, param_distributions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV with ratings")
    ap.add_argument("--excel", type=str, default=None, help="Path to Excel with ratings")
    ap.add_argument("--sheet", type=str, default=None, help="Excel sheet name")
    ap.add_argument("--target", type=str, default="mos", choices=["mos", "dmos_100"], help="Train target")
    ap.add_argument("--use_objective", type=int, default=1, help="Use VMAF/SSIM/PSNR if available (1/0)")
    ap.add_argument("--model", type=str, default="hgb", choices=["hgb", "xgb"], help="Regressor")
    ap.add_argument("--cv_splits", type=int, default=5, help="GroupKFold splits")
    ap.add_argument("--group_by", type=str, default="video_name", help="Group column for CV & duplicate aggregation")
    ap.add_argument("--out", type=str, default="quality_model.joblib", help="Output model path")
    args = ap.parse_args()

    # Load
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.excel:
        df = pd.read_excel(args.excel, sheet_name=args.sheet or 0)
    else:
        raise ValueError("Provide --csv or --excel")

    # Basic normalization of column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Safety: expected columns
    required_basic = {"video_name", "character", "view", "compression", "qp"}
    missing = required_basic - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute MOS (from components if present), MOS_ref, DMOS_100
    df = compute_mos_cols(df)
    if "mos" not in df.columns:
        raise ValueError("No MOS columns present / derivable.")
    df = compute_mosref_and_dmos(df)

    # Drop obviously invalid rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mos", "mos_ref", "dmos_100", "qp", "character", "view", "compression"])

    # Aggregate duplicates per video (average multiple ratings for same clip)
    if args.group_by in df.columns:
        df = aggregate_duplicates(df, args.group_by)

    # Features
    cat_cols = ["character", "view", "compression"]
    num_cols = ["qp"]

    # Optional objective metrics
    if args.use_objective:
        for col in ["vmaf", "ssim", "psnr"]:
            if col in df.columns:
                num_cols.append(col)

    # Light interactions
    df["qp2"] = df["qp"] ** 2
    num_cols.append("qp2")

    # Choose target
    target_col = args.target  # "mos" or "dmos_100"
    y = df[target_col].astype(float).values

    X = df[cat_cols + num_cols].copy()

    # Preprocess: One-hot for categoricals, passthrough numeric
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )

    # Model & small randomized search
    base_model, param_dists = build_model(args.model)

    pipe = Pipeline([
        ("pre", pre),
        ("model", base_model),
    ])

    # GroupKFold
    if args.group_by in df.columns:
        groups = df[args.group_by].astype(str).values
    else:
        # fallback: group by (character, view) to avoid trivial leakage
        groups = (df["character"].astype(str) + "_" + df["view"].astype(str)).values

    gkf = GroupKFold(n_splits=args.cv_splits)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dists,
        n_iter=20,
        cv=gkf.split(X, y, groups=groups),
        verbose=1,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        random_state=42,
        refit=True,
    )

    search.fit(X, y)

    print("\nBest params:", search.best_params_)
    best_pipe = search.best_estimator_

    # Manual CV evaluation with the best params (honest reporting)
    preds_all = np.zeros_like(y, dtype=float)
    fold_scores = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va, y_va = X.iloc[va], y[va]

        # Refit a clone of best_pipe on this fold
        pipe_fold = Pipeline([
            ("pre", pre),
            ("model", type(best_pipe.named_steps["model"])(**{
                k.replace("model__", ""): v
                for k, v in search.best_params_.items()
                if k.startswith("model__")
            }))
        ])
        pipe_fold.fit(X_tr, y_tr)
        y_hat = pipe_fold.predict(X_va)
        fold_scores.append(eval_regression(y_va, y_hat, label=f"CV fold {fold}"))
        preds_all[va] = y_hat

    # Overall CV metrics
    print()
    eval_regression(y, preds_all, label="CV overall")

    # Fit final on all data and save
    best_pipe.fit(X, y)
    dump({
        "pipeline": best_pipe,
        "target": target_col,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "best_params": search.best_params_
    }, args.out)

    print(f"\nSaved model to: {args.out}")
    print("Use the saved pipeline to predict pMOS/DMOS_100 for your 8,448 videos.")


if __name__ == "__main__":
    main()
