import argparse, math
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.svm import SVR

def read_excel_safely(path, sheet_arg):
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

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def metrics(y_true, y_pred, tag="CV"):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0] if len(y_true)>1 else np.nan
    spea = spearmanr(y_true, y_pred)[0] if len(y_true)>1 else np.nan
    print(f"[{tag}] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Pearson={pear:.4f} | Spearman={spea:.4f}")
    return rmse, mae, r2, pear, spea

def build_pipeline(model_name, cat_cols, num_cols):
    pre = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )

    if model_name == "linear":
        model = LinearRegression()
        return Pipeline([("pre", pre), ("model", model)])

    if model_name == "elasticnet":
        # Standardize after OHE (dense)
        scaler = StandardScaler()
        en = ElasticNetCV(
            l1_ratio=[0.0,0.1,0.25,0.5,0.75,0.9,1.0],
            alphas=None,  # auto path
            cv=5,
            random_state=42
        )
        return Pipeline([("pre", pre), ("scaler", scaler), ("model", en)])

    if model_name == "svr":
        scaler = StandardScaler()
        svr = SVR(C=10.0, epsilon=0.1, gamma="scale", kernel="rbf")
        return Pipeline([("pre", pre), ("scaler", scaler), ("model", svr)])

    raise ValueError("Unknown --model. Choose from: linear, elasticnet, svr.")

def main():
    ap = argparse.ArgumentParser(description="Proxy regression on 8,448 using pMOS as target")
    ap.add_argument("--xlsx", required=True, help="all_8448_with_pMOS.xlsx")
    ap.add_argument("--sheet", default=None, help="sheet name or index (0 = first)")
    ap.add_argument("--model", required=True, choices=["linear","elasticnet","svr"])
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--out_csv", default=None, help="optional: save OOF preds CSV")
    args = ap.parse_args()

    df = read_excel_safely(args.xlsx, args.sheet)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required cols
    needed = {"video_name","character","view","compression","qp","pmos"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}. Got: {list(df.columns)}")

    # Numerics
    for c in ["qp","vmaf","ssim","psnr","pmos"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    # Derived
    df["qp2"] = df["qp"]**2

    cat_cols = ["character","view","compression"]
    num_cols = ["qp","qp2"] + [c for c in ["vmaf","ssim","psnr"] if c in df.columns]

    # Drop missing rows
    df = df.dropna(subset=cat_cols + num_cols + ["pmos"])

    X = df[cat_cols + num_cols].copy()
    y = df["pmos"].astype(float).values
    # Group by character+view (prevents trivial overlap)
    groups = (df["character"].astype(str) + "_" + df["view"].astype(str)).values

    pipe = build_pipeline(args.model, cat_cols, num_cols)

    # OOF CV
    gkf = GroupKFold(n_splits=args.cv_splits)
    from sklearn.base import clone
    y_oof = np.zeros_like(y, dtype=float)

    for i,(tr,va) in enumerate(gkf.split(X, y, groups=groups), 1):
        p = clone(pipe)
        p.fit(X.iloc[tr], y[tr])
        y_hat = p.predict(X.iloc[va])
        metrics(y[va], y_hat, tag=f"{args.model.upper()} fold {i}")
        y_oof[va] = y_hat

    metrics(y, y_oof, tag=f"{args.model.upper()} overall")

    if args.out_csv:
        out = df[["video_name","character","view","compression","qp"]].copy()
        out["pmos"] = y
        out["pmos_pred"] = y_oof
        out.to_csv(args.out_csv, index=False)
        print(f"Saved OOF predictions: {args.out_csv}")

if __name__ == "__main__":
    main()
