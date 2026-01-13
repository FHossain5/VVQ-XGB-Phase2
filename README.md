# XGBoost pMOS Model

## Abstract
This report presents an XGBoost regression model trained on the full catalogue of 8,448 volumetric videos for predicted Mean Opinion Score (pMOS) estimation. The objective of this model is not to measure human perception directly, but to provide a numerically consistent, feature-space aligned predictor across compression levels, characters, and viewpoints. All results are reported on the 1–5 pMOS scale, with errors (RMSE, MAE) and correlations (Pearson, Spearman) reflecting regression consistency rather than direct subjective quality assessment.

---

## 1. Methodology

### 1.1 Dataset and Features
The dataset used is **all_8448_with_pMOS.xlsx**, containing volumetric video metadata, compression settings, and pMOS values.

The following features were used as input variables:

- **Categorical:** character, view, compression  
- **Numerical:** qp, qp², vmaf, ssim, psnr  

The addition of **qp² (quantization parameter squared)** captures the non-linear effect of compression on perceived quality.

The data was preprocessed by:
- Lowercasing column names  
- Type conversion to numeric (for QP, VMAF, SSIM, PSNR, and pMOS)  
- Global NaN removal (`df.dropna().reset_index(drop=True)`)  

After preprocessing, **8,448 valid samples** remained.

---

### 1.2 Model Design
The final model uses **XGBoost (tree_method='hist')**, chosen for its stability, explainability, and superior handling of non-linear relationships. The pipeline includes preprocessing via **Column Transformer** for one-hot encoding of categorical columns and passthrough of numerical columns.

#### Final configuration

| Parameter | Value |
|---|---:|
| learning_rate | 0.09 |
| max_depth | 4 |
| min_child_weight | 5 |
| n_estimators | 300 |
| subsample | 0.7 |
| colsample_bytree | 1.0 |
| reg_lambda | 5.0 |
| random_state | 42 |
| objective | reg:squarederror |

Reproducibility was ensured by fixing **random_state = 42** and using deterministic **GroupKFold** splits.

---

### How pMOS Was Obtained
The pMOS values used in this work were generated in two stages: real human subjective ratings followed by model-based generalisation. First, a controlled subjective experiment was conducted on the base set of **240 volumetric videos**, with each video being viewed, in a view-and-rate format, multiple times by different human subjects, following the **ITU-T P.910** recommendations on conducting subjective tests. Individual opinion scores were cleaned using standard outlier-rejection procedures and the remaining valid responses averaged to yield a reliable Mean Opinion Score (MOS) on the 1–5 quality scale. These are its only part of the dataset that contain data from genuine human quality judgements.

The second stage was to take these **240 MOS labelled videos** and train a neural regression model (**EfficientNet-B3**). This model learns to predict subjective quality from objective and metadata-based features. Once this model was trained, it was then applied to our original full collection of **8,448** compressed video versions, again with all four QP levels (15, 25, 35, 45) and with all compression types. Since human MOS existed only for the original 240 videos, the other ~97% of the dataset needed model-based estimation. The predicted scores produced from this model, then, become the pMOS values we use in the XGBoost regression study. pMOS thus allows us to extend the subjective scale for all compression settings, firmly anchored to real human judgements.

---

## Fold-Wise Performance Table
Fold-wise evaluation results from the 5-fold GroupKFold cross-validation protocol, showing the model’s consistency across all splits in terms of explained variance (R²), prediction error (RMSE, MAE), and both linear (Pearson r) and rank-order (Spearman ρ) correlations between predicted and true pMOS values.

---

## 2. Evaluation Protocol

### 2.1 Cross-Validation
We employed **GroupKFold (n_splits = 5)**, grouping by **(character + view)** to prevent overlap between folds of similar viewpoints of the same character. Each fold acts as an unseen test set, ensuring a fair Out-of-Fold (OOF) evaluation.

### 2.2 Metrics
The following performance indicators were computed:

- **R² (Coefficient of Determination)**  
- **RMSE (Root Mean Square Error)**  
- **MAE (Mean Absolute Error)**  
- **Pearson r** — linear correlation coefficient  
- **Spearman ρ** — rank correlation  

All metrics are reported on the **1–5 pMOS scale**.

---

## 3. Results and Visuals

### 3.1 Fold-Wise Performance

| Fold | R² | RMSE | MAE | Pearson r | Spearman ρ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9953 | 0.0459 | 0.0283 | 0.9977 | 0.9960 |
| 2 | 0.9965 | 0.0372 | 0.0251 | 0.9983 | 0.9955 |
| 3 | 0.9977 | 0.0327 | 0.0180 | 0.9990 | 0.9948 |
| 4 | 0.9951 | 0.0456 | 0.0340 | 0.9977 | 0.9917 |
| 5 | 0.9959 | 0.0408 | 0.0290 | 0.9980 | 0.9960 |

#### Overall Results (OOF)

| Metric | Value |
|---|---:|
| R² | 0.9961 |
| RMSE | 0.0411 |
| MAE | 0.0270 |
| Pearson r | 0.9980 |
| Spearman ρ | 0.9952 |
| R² mean ± std | 0.9961 ± 0.0010 |

These metrics demonstrate near-perfect alignment between predicted and true pMOS values. Errors are within ±0.04 on a 1–5 scale, confirming highly consistent performance across all folds.

---

## Calibration and Confidence Interval Analysis
To verify that the XGBoost model is not only accurate but also well aligned with the pMOS scale, we applied a simple linear calibration on the out-of-fold (OOF) predictions. This correction is intentionally light-weight. It adjusts for any tiny global bias while preserving the ranking and relative structure of the predictions.

After calibration, the performance remains essentially unchanged, with a very small improvement in RMSE, confirming that the original model was already very well aligned with the target:

| Metric | Before Calibration | After Calibration |
|---|---:|---:|
| R² | 0.9961 | 0.9961 |
| RMSE | 0.0411 | 0.0410 |
| PLCC | 0.9980 | 0.9980 |
| SRCC | 0.9952 | 0.9952 |

The difference is numerically minor, which is exactly what we want. It shows that the XGBoost model was not drifting or systematically biased, and the calibration step only fine tunes the scale in a statistically clean way.

To quantify robustness, we computed 95% bootstrap confidence intervals (2,000 resamples) on the calibrated predictions:

| Metric | 95% Confidence Interval (Bootstrap) |
|---|---|
| R² | 0.9958 – 0.9963 |
| PLCC | 0.9979 – 0.9982 |
| SRCC | 0.9948 – 0.9955 |

These very tight intervals confirm that the model’s consistency is not accidental. The behavior is stable across resamples and supports the reliability of the reported numbers.

---

### 3.2 Visualizations
The following figures illustrate fold consistency and model interpretability.

**Where to add images:** place these PNG files inside `xgb_results/` in your repository (same names). Then keep the Markdown links below as-is.

- Figure 1: R² by Fold (stable predictive strength across all folds).  
  ![R² by fold](xgb_results/xgb_r2_by_fold.png)

- Figure 2: RMSE by Fold ( minimal error variation).  
  ![RMSE by fold](xgb_results/xgb_rmse_by_fold.png)

- Figure 3: Fold-wise Pearson Correlation for the XGBoost pMOS Model  
  ![Pearson by fold](xgb_results/xgb_pearson_by_fold.png)

- Figure 4: Fold-wise spearman Correlation for the XGBoost pMOS Model  
  ![Spearman by fold](xgb_results/xgb_spearman_by_fold.png)

- Figure 5: OOF Scatter Plot — predicted vs. true pMOS  
  ![OOF scatter uncalibrated](xgb_results/xgb_oof_scatter_uncalibrated.png)

- Figure 6: Top Ten Most Influential Features Identified by the XGBoost pMOS Model  
  ![Top-10 feature importance](xgb_results/xgb_feature_importance_top10.png)

- Figure 7: Uncalibrated Scatter (True vs Predicted pMOS)  
  ![Uncalibrated scatter](xgb_results/xgb_oof_scatter_uncalibrated.png)

- Figure 8: Calibrated Scatter (True vs Predicted pMOS)  
  ![Calibrated scatter](xgb_results/xgb_oof_scatter_calibrated.png)

- Figure 9: Calibration Curves (Uncalibrated vs Calibrated)  
  ![Calibration curves](xgb_results/xgb_calibration_curves.png)

---

## 4. Analysis

### 4.1 Interpretation
The model explains 99.61% of the variance in pMOS, confirming that volumetric video quality (as measured by objective metrics like VMAF, SSIM, and PSNR) is captured effectively by the feature set.

Key insights:
- VMAF, SSIM, and QP dominate feature importance, aligning with perceptual understanding of compression.
- Low RMSE and MAE show minimal deviation between predictions and target.
- Inter-fold R² variance (±0.0010) indicates stable generalization.
- OOF scatter confirms the absence of systemic bias — predictions cluster tightly around the y=x diagonal.
- Feature importance analysis reveals that:
  - Higher VMAF and SSIM correlate strongly with higher predicted quality.
  - QP and QP² show an inverse, nonlinear relationship with pMOS.
  - Compression type contributes modestly but consistently, refining model calibration per encoder type.

