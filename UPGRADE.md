# Corrected TVT experiment and next steps

## What was wrong

The legacy training script sets y=DTW_TVT and includes TVT and training-only formation markers in X. It therefore trains a surrogate alignment predictor using unavailable inputs, not the stated TVT predictor. Unrestricted dropna removes the post-PS rows where TVT_input is deliberately missing. Random row splitting mixes observations from the same well, and RobustScaler is fitted before splitting. The published 70.8039 score is not comparable with the corrected experiment.

The current FT-Transformer treats columns as tokens within a row. It has no along-well temporal attention. Statements that it learns trajectory sequences or that trees cannot learn feature interactions are unsupported.

## Implemented experiment

`train_residual.py` learns `TVT - last_known_TVT_input` and reconstructs absolute TVT. It uses an explicit feature allowlist, relative coordinates, trajectory gradients, trailing GR statistics, lags, the known-prefix trend and distance from PS. It preserves missing GR and excludes actual TVT, formation markers, and legacy DTW from model features. Train and inference use the same feature builder.

A deterministic well-group split assigns approximately 60%/20%/20% of wells to training/validation/test. Two histogram gradient boosting configurations compete against constant-anchor and prefix-linear baselines on validation RMSE. The winner is refitted on train+validation, then evaluated on the untouched test wells. No random-row internal early stopping is used. Both pooled row RMSE and per-well metrics are saved. Model and split identifiers are saved for reproducibility.

The test result is a final estimate for this fixed experiment. Do not repeatedly choose changes using it; reserve a fresh external test set or use nested grouped cross-validation for further model selection. For correlated neighboring wells, also test geographic/pad holdouts when metadata is available.

## Run

From this repository folder:

```bash
pip install -r requirements_baseline.txt
python -m unittest discover -p test_residual.py -v
python train_residual.py train --data "C:/Users/haris/Downloads/Harish/Hackathon/train/MASTER_training_data_v2.csv" --output runs/residual
python train_residual.py predict --data test_master.csv --model runs/residual/model.joblib --output predictions.csv
```

The existing master CSV may be used: legacy DTW and other extra columns are ignored. Training requires WELL_ID, MD, X, Y, Z, GR, TVT_input and ground-truth TVT. Inference needs the same fields except TVT. Each well must include its known prefix and unknown suffix, with TVT_input present only in the prefix. The script sorts by MD, requires unique finite MD values and rejects ambiguous masks, missing anchors and wells without prediction rows. Missing sentinel values such as -999 must first be converted according to the dataset specification; do not guess sentinel values. Relative coordinates require valid anchor coordinates; review missingness before interpreting performance.

Outputs: `metrics.json`, `test_predictions.csv`, and `model.joblib`. Predictions retain source_row (zero-based original CSV row), WELL_ID and MD. This is not automatically a competition submission: join against the official sample submission and validate its ID/order rules first. Only load trusted joblib files.

The GR windows are trailing row-count windows, not fixed physical-distance windows. If MD sampling intervals vary materially, evaluate fixed-distance resampling/windows. No typewell-derived feature is used in this first corrected baseline. No claim of improved real-data RMSE is made: the dataset is absent from the repository. Six safety tests and a synthetic end-to-end train/save/load/predict check passed; synthetic scores are not geological performance evidence.

## Best next experiment, conditional on the data

1. Establish the corrected baseline, prefix quality, and error versus distance from PS. Investigate the worst wells and discontinuities rather than just pooled RMSE.
2. Build an anchor-conditioned typewell matching feature. Current full FastDTW forces endpoint-to-endpoint monotone alignment and does not condition on known TVT_input. A horizontal well may revisit stratigraphy, so monotone traversal of the typewell must not be assumed universally. First verify the TVT convention, geometry and competition input availability. Compare an anchored local/subsequence alignment or a state-space tracker allowing stationary, positive and negative TVT steps where supported. Tune transition penalties on training/validation wells only.
3. For streaming use, alignment must use only GR and survey data observed up to the current point. Offline access to the whole unknown-zone GR sequence is a different task and must be labelled and evaluated separately. The baseline supplied here uses only current/past observations.
4. Compare CatBoost/LightGBM residual regression with the supplied histogram booster on identical grouped folds. Add anchor-corrected alignment, matching cost/ambiguity, multi-scale GR and geometry features through ablations. Do not include unobserved target or formation labels.
5. Try a causal TCN/sequence model only if residual analysis shows useful sequential structure the engineered features miss. A larger Transformer is not evidence of a better approach. Retain an ensemble only when independently validated.

There is no defensible perfect-model promise before these experiments. The best justified direction is known-prefix anchoring plus validated signal matching and residual regression, with model complexity earned by held-out improvements.

## Information needed for real benchmarking

Provide the training horizontal/typewell CSV pairs (preferably all wells), official dataset/competition link, test schema and sample submission, TVT/PS definitions, any rules on future GR availability, and available RAM/GPU. At least five wells are needed to execute this split, but a tiny sample cannot establish generalization. Original raw CSV pairs are needed to repair and evaluate alignment. Do not share API keys or credentials.

References:
- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/cross_validation.html
- https://catboost.ai/docs/en/concepts/loss-functions-regression
