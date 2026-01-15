import argparse
import pandas as pd
import numpy as np
from joblib import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default="0")
    ap.add_argument("--out", default="all_8448_with_pMOS.xlsx")
    ap.add_argument("--fmt", choices=["xlsx","csv"], default="xlsx")
    args = ap.parse_args()

    # Load model bundle & expected features
    bundle = load(args.model)
    pipe = bundle["pipeline"]
    cat_cols = bundle["cat_cols"]
    num_cols = bundle["num_cols"]

    # Read Excel (single sheet)
    sheet = args.sheet
    try: sheet = int(sheet)
    except: pass
    df = pd.read_excel(args.excel, sheet_name=sheet)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize compression to match training
    if "compression" in df.columns:
        df["compression"] = df["compression"].astype(str).str.strip().str.lower()
        comp_map = {
            "geometry compression": "GC",
            "geometry-compression": "GC",
            "gc": "GC",
            "tc": "TC",
            "texture compression": "TC",
            "gtc": "GTC",
            "geometry+texture compression": "GTC",
        }
        df["compression"] = df["compression"].map(comp_map).fillna(df["compression"])

    # Ensure numerics
    if "qp" in df.columns:
        df["qp"] = pd.to_numeric(df["qp"], errors="coerce")
    for c in ["vmaf","ssim","psnr"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Create qp2 if model uses it
    if "qp2" in num_cols and "qp2" not in df.columns:
        df["qp2"] = df["qp"] ** 2

    # ---------- IMPUTE MISSING FEATURES (per (character,view,compression,qp) mean → global mean) ----------
    feature_cols_all = [c for c in (cat_cols + num_cols)]
    present_feats = [c for c in feature_cols_all if c in df.columns]

    combo = ["character","view","compression","qp"]
    num_feats = [c for c in num_cols if c in ["vmaf","ssim","psnr","qp2","qp"] and c in df.columns]

    if all(c in df.columns for c in combo) and num_feats:
        means = df.groupby(combo)[num_feats].transform("mean")
        for c in num_feats:
            df[c] = df[c].fillna(means[c])

    # Global fallback means
    for c in num_feats:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())

    # For categoricals, fill NAs with "unknown"
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")
    # ----------------------------------------------------------------------------------

    # Final feature set actually available
    feature_cols = [c for c in (cat_cols + num_cols) if c in df.columns]

    # Predict pMOS
    X = df[feature_cols]
    pmos = pipe.predict(X)
    df["pMOS"] = np.clip(pmos, 1.0, 5.0)

    # DMOS_100 from predicted MOS within (character, view)
    if all(c in df.columns for c in ["character","view"]):
        df["mos_ref_pred"] = df.groupby(["character","view"])["pMOS"].transform("max")
        df["dmos_100_pred"] = 25.0 * (df["mos_ref_pred"] - df["pMOS"])

    # Save
    if args.fmt == "csv":
        df.to_csv(args.out, index=False)
    else:
        df.to_excel(args.out, index=False)

    print(f"Saved: {args.out}  | rows: {len(df)}")
    print(df[["video_name","character","view","compression","qp","pMOS","dmos_100_pred"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
