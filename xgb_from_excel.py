
import argparse, math, json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from joblib import dump
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def read_excel_safely(path, sheet_arg):
    print(f"[INFO] Loading Excel: {path}")
    if sheet_arg is not None:
        try:
            df = pd.read_excel(path, sheet_name=int(sheet_arg)); print(f"[INFO] Loaded sheet index: {sheet_arg}"); return df
        except (ValueError, TypeError): pass
        try:
            df = pd.read_excel(path, sheet_name=sheet_arg); print(f"[INFO] Loaded sheet name: {sheet_arg}"); return df
        except Exception as e:
            print(f"[WARN] Sheet '{sheet_arg}' not found ({e}). Falling back to first sheet.")
    xls = pd.ExcelFile(path); print(f"[INFO] Available sheets: {xls.sheet_names}. Using first.")
    return pd.read_excel(path, sheet_name=0)

def eval_metrics(y_true, y_pred, tag="CV"):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    spea = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    print(f"[{tag}] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Pearson={pear:.4f} | Spearman={spea:.4f}")
    return {"RMSE":rmse, "MAE":mae, "R2":r2, "Pearson":pear, "Spearman":spea}

def save_fold_plots(folds, prefix="xgb_pmos"):
    fold_ids = [f["fold"] for f in folds]
    r2_vals  = [f["R2"] for f in folds]
    rmse_vals= [f["RMSE"] for f in folds]
    pear_vals= [f["Pearson"] for f in folds]

    plt.figure(); plt.bar(fold_ids, r2_vals); plt.title("R² by Fold"); plt.xlabel("Fold"); plt.ylabel("R²"); plt.ylim(0,1); plt.tight_layout()
    plt.savefig(f"{prefix}_r2_by_fold.png", dpi=300); plt.close()

    plt.figure(); plt.bar(fold_ids, rmse_vals); plt.title("RMSE by Fold"); plt.xlabel("Fold"); plt.ylabel("RMSE"); plt.tight_layout()
    plt.savefig(f"{prefix}_rmse_by_fold.png", dpi=300); plt.close()

    plt.figure(); plt.bar(fold_ids, pear_vals); plt.title("Pearson by Fold"); plt.xlabel("Fold"); plt.ylabel("Pearson r"); plt.ylim(0,1); plt.tight_layout()
    plt.savefig(f"{prefix}_pearson_by_fold.png", dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Train XGB on 8,448 Excel using pMOS as labels (consistency check).")
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--target", default="pmos")
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--out", default="pMOS_xgb.joblib")
    ap.add_argument("--save_oof", default="oof_pmos_8448.csv")
    args = ap.parse_args()

    print("NOTE: Model is trained on pMOS (predicted MOS), not direct human MOS. "
          "These results measure regression consistency and feature-space alignment at scale.")
    try:
        df = read_excel_safely(args.excel, args.sheet)
        df.columns = [c.strip().lower() for c in df.columns]
        print(f"[INFO] Rows loaded: {len(df)}")

        needed = {"video_name","character","view","compression","qp", args.target}
        miss = needed - set(df.columns)
        if miss: raise ValueError(f"Missing columns: {miss}")

        for c in ["qp","vmaf","ssim","psnr",args.target]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["qp2"] = df["qp"]**2

        cat_cols = ["character","view","compression"]
        num_cols = ["qp","qp2"] + [c for c in ["vmaf","ssim","psnr"] if c in df.columns]
        feat_cols = cat_cols + num_cols

        before = len(df)
        df = df.dropna(subset=feat_cols + [args.target])
        print(f"[INFO] Dropped {before-len(df)} rows with NA (kept {len(df)})")

        X = df[feat_cols].copy()
        y = df[args.target].astype(float).values
        groups = (df["character"].astype(str)+"_"+df["view"].astype(str)).values  # deterministic grouping

        pre = ColumnTransformer([("cat", make_ohe(), cat_cols), ("num", "passthrough", num_cols)], remainder="drop")

        xgb = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", random_state=42,
            n_estimators=300, learning_rate=0.07, max_depth=4,
            subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0
        )
        pipe = Pipeline([("pre", pre), ("model", xgb)])

        param_distributions = {
            "model__n_estimators":[200,300,400,600],
            "model__learning_rate":[0.05,0.07,0.09],
            "model__max_depth":[3,4,5,6],
            "model__subsample":[0.7,0.8,0.9,1.0],
            "model__colsample_bytree":[0.8,1.0],
            "model__reg_lambda":[0.1,1.0,5.0],
            "model__min_child_weight":[1,5,10],
        }

        gkf = GroupKFold(n_splits=args.cv_splits)  # deterministic (no shuffle)

        search = RandomizedSearchCV(
            estimator=pipe, param_distributions=param_distributions, n_iter=25,
            cv=gkf.split(X,y,groups=groups), scoring="neg_mean_squared_error",
            n_jobs=-1, random_state=42, refit=True, verbose=1
        )

        print("[INFO] Fitting randomized search… (errors are on 1–5 pMOS scale)")
        search.fit(X,y)
        print("[INFO] Best params:", search.best_params_)

        # OOF metrics
        from sklearn.base import clone
        y_oof = np.zeros_like(y, dtype=float)
        fold_metrics = []
        for i,(tr,va) in enumerate(gkf.split(X,y,groups=groups),1):
            mdl = clone(search.best_estimator_)
            mdl.fit(X.iloc[tr], y[tr])
            y_hat = mdl.predict(X.iloc[va])
            m = eval_metrics(y[va], y_hat, tag=f"OOF fold {i}")
            fold_metrics.append({"fold":i, **m})
            y_oof[va] = y_hat

        overall = eval_metrics(y, y_oof, tag="OOF overall")
        r2_mean = float(np.mean([f["R2"] for f in fold_metrics]))
        r2_std  = float(np.std([f["R2"] for f in fold_metrics]))
        print(f"[INFO] Inter-fold R² mean ± std: {r2_mean:.4f} ± {r2_std:.4f}")

        # Save model & OOF
        from joblib import dump
        dump(search.best_estimator_, args.out); print(f"[INFO] Saved model -> {args.out}")
        oof = df[["video_name","character","view","compression","qp"]].copy()
        oof["y_true"] = y; oof["y_pred"] = y_oof
        oof.to_csv(args.save_oof, index=False); print(f"[INFO] Saved OOF -> {args.save_oof}")

        # Save metrics JSON
        with open("student_xgb_oof_metrics.json","w") as f:
            json.dump({"overall":overall, "folds":fold_metrics, "best_params":search.best_params_,
                       "r2_mean":r2_mean, "r2_std":r2_std}, f, indent=2)
        print("[INFO] Saved metrics -> student_xgb_oof_metrics.json")

        # Feature importances (top-10)
        mdl = search.best_estimator_.named_steps["model"]
        pre = search.best_estimator_.named_steps["pre"]
        try:
            feat_names = pre.get_feature_names_out(cat_cols + num_cols)
        except Exception:
            feat_names = np.array([f"f{i}" for i in range(mdl.n_features_in_)])
        fi = mdl.feature_importances_
        idx = np.argsort(fi)[::-1][:10]
        plt.figure()
        plt.bar(range(len(idx)), fi[idx])
        plt.xticks(range(len(idx)), [str(feat_names[i]) for i in idx], rotation=45, ha="right")
        plt.title("Top 10 Most Important Features (XGB pMOS)")
        plt.tight_layout()
        plt.savefig("xgb_feature_importance.png", dpi=300)
        print("[INFO] Saved xgb_feature_importance.png")

        # Fold charts
        save_fold_plots(fold_metrics, prefix="xgb_pmos")
        print("[DONE] Training + artifacts complete.")

    except Exception as e:
        print(f"[ERROR] {e}"); sys.exit(1)

if __name__ == "__main__":
    main()
