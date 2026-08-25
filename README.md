# ROGII - Geology Wellbore Prediction-
This repository is dedicated to organizing and maintaining my project work, code, datasets, and documentation.
# GeoSteer-AI: Subsurface Geological Trajectory & TVT Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastDTW](https://img.shields.io/badge/Signal-FastDTW-brightgreen.svg)](https://github.com/slaypni/fastdtw)

A hybrid deep-learning and signal-alignment pipeline engineered to predict **True Vertical Thickness (TVT)** across evaluation zones in horizontal well trajectories using Dynamic Time Warping (FastDTW) and Feature-Tokenizer Transformers (FT-Transformer).

---

## 📌 Problem Overview

In directional drilling and geosteering workflows, tracking accurate stratigraphic depth—measured via True Vertical Thickness (TVT)—is critical to maintaining the wellbore inside optimal formation pay-zones. Beyond the Prediction Start (PS) point, target labels are unobserved, requiring models to learn non-linear spatial dips and cross-correlate lateral Gamma Ray (GR) measurements with vertical Typewell reference logs.

```text
  Surface
     │
     │ (Vertical Section)
     └──┐
        │  Horizontal Trajectory (MD, X, Y, Z, GR)
        └───────────────────────────────────────────► [PS Point] ──► Predicted TVT (Evaluation Zone)
                     ▲                                                    │
                     │ (Dynamic Time Warping)                             ▼
        Typewell Log (TVT, GR, Geology) ──────────────────────────► FT-Transformer Target

```
## 🏗️ Architecture & Pipeline

-  Signal Alignment (FastDTW): Computes non-linear pairwise warping paths between lateral Gamma Ray (GR) response signals and vertical Typewell reference logs.

-  Stratigraphic Feature Engineering: Transposes matched vertical typewell depths into horizontal step sequences (DTW_TVT) and couples them with 3D spatial coordinates (MD, X, Y, Z).

-  Robust Preprocessing: Applies RobustScaler across continuous features to minimize the influence of extreme log spikes and outliers.

-  Deep Tabular Modeling: Employs an FT-Transformer (Feature Tokenizer + multi-layer Transformer Encoder) to capture complex non-linear feature interactions.

-  Loss Function: Trains directly against Mean Squared Error (MSE) loss to optimize for Root Mean Squared Error (RMSE) performance.

## 📊 Dataset Schema

| Dataset Source | Feature Column | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **Horizontal Well**[cite: 1] | `MD` | `float32` | Measured Depth (ft) along the lateral wellbore from surface.[cite: 1] |
| | `X`, `Y` | `float32` | Spatial Easting and Northing coordinates (ft).[cite: 1] |
| | `Z` | `float32` | True Vertical Depth below sea level (ft).[cite: 1] |
| | `GR` | `float32` | Gamma Ray log measuring natural formation radioactivity (API).[cite: 1] |
| | `TVT_input` | `float32` | TVT feature available prior to the Prediction Start (PS) point.[cite: 1, 7] |
| | `TVT` | `float32` | Ground-truth True Vertical Thickness (ft) [Target; train only].[cite: 1] |
| | `ANCC` ... `BUDA` | `float32` | Stratigraphic formation boundary markers [Train only].[cite: 1] |
| **Typewell**[cite: 1] | `TVT` | `float32` | Vertical Depth Index (ft) correlating to lateral geological position.[cite: 1] |
| | `GR` | `float32` | Vertical Gamma Ray reference signature used for log correlation.[cite: 1] |
| | `Geology` | `string` | Categorical formation label (e.g., ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA).[cite: 1] |

## 🚀 Quickstart
1. Installation
```
git clone https://github.com/your-username/GeoSteer-AI.git
cd GeoSteer-AI
pip install -r requirements.txt
```
Requirements: torch>=2.0.0, pandas>=2.0.0, numpy>=1.24.0, scikit-learn>=1.2.0, scipy>=1.10.0, fastdtw>=0.3.4[cite: 2, 4, 5].

2. Feature Extraction with Dynamic Time Warping
Process raw lateral and vertical logs to compute DTW mappings:

```
python src/dtw_alignment.py --data_dir data/train/ --output data/MASTER_training_data_v2.csv
```

3. Model Training & Evaluation
Train the FT-Transformer on aligned tabular features:
```
python src/train.py --data_path data/MASTER_training_data_v2.csv --epochs 50 --batch
```

## 📈 Evaluation Metric
Model performance is evaluated using Root Mean Squared Error (RMSE) across all prediction coordinates beyond the Prediction Start (PS) index[cite: 2, 7]:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\text{TVT}_{\text{actual}}^{(i)} - \text{TVT}_{\text{pred}}^{(i)}\right)^2}$$

  [cite: 2, 7]
## 🤝 Contributing & License
Contributions are welcome. Please open an issue or submit a pull request for improvements in feature extraction, stratigraphic modeling, or model architectures.Distributed under the MIT License. See LICENSE for more information.
