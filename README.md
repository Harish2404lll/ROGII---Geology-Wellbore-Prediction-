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
## 🏹 System Architecture: 

<img width="1390" height="669" alt="image" src="https://github.com/user-attachments/assets/b57beadb-280c-4fa1-a5b3-536372706ea9" />

## 📈 Evaluation Metric
Model performance is evaluated using Root Mean Squared Error (RMSE) across all prediction coordinates beyond the Prediction Start (PS) index[cite: 2, 7]:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\text{TVT}_{\text{actual}}^{(i)} - \text{TVT}_{\text{pred}}^{(i)}\right)^2}$$

  [cite: 2, 7]

## 📊 Experimental Results & Model Performance

The `FT-Transformer` architecture was evaluated using an 80/20 train/test split on the master aligned dataset across 49 epochs using the `RobustScaler` and `MSELoss` optimization setup[cite: 1]:

| Metric | Target | Final Score |
| :--- | :--- | :--- |
| **Training RMSE Loss**[cite: 1] | Stratigraphic Thickness ($TVT$)[cite: 1, 7] | **81.9368 ft** |
| **Final Test Target MAE**[cite: 1] | Stratigraphic Thickness ($TVT$)[cite: 1, 7] | **45.9464 ft** |
| **Final Test Target RMSE**[cite: 1] | Stratigraphic Thickness ($TVT$)[cite: 1, 7] | **70.8039 ft** |

### Convergence History (Final Epochs)


<img width="1120" height="409" alt="image" src="https://github.com/user-attachments/assets/85e7edac-cf78-4faa-9935-c221c25f6d64" />


## 📚 Related Work

Subsurface interpretation and automated geosteering traditionally rely on deterministic algorithms or manual workflows across three main areas:

* **Signal Matching & Dynamic Programming (DTW, Hale's Warping, HMMs):** Effective for 1D vertical alignments, but brittle under structural folding, faulting, and severe lateral stretch-squeeze. *Our approach* uses DTW strictly as a downstream feature extractor (`DTW_TVT`) rather than an unguided heuristic decoder.
* **Machine Learning for Horizon Tracking (GBDTs, PINNs):** Standard tree models treat survey points independently, missing inter-feature dependencies. *Our approach* deploys **FT-Transformers** with self-attention to capture complex multi-scale tabular interactions.
* **Spatial Interpolation (Kriging & Variography):** Fails in variable-dip basins with sparse offset control. *Our approach* conditions predictions on live MWD Gamma Ray and trajectory kinematics ($X, Y, Z, MD$) for dynamic responsiveness.

---

## 🔮 Future Work Roadmap

* **[P1] Geological Loss Constraints:** Penalize physical dip violations and unrealistic $\frac{d\text{TVT}}{d\text{MD}}$ gradients.
* **[P2] Offset-Well Graph Attention (GNN):** Enable dynamic spatial conditioning across multi-well pads.
* **[P3] CNN–Transformer Hybrid:** Add multi-scale 1D temporal convolutions over local Gamma Ray windows.
* **[P4] Uncertainty Quantification:** Implement Bayesian Monte Carlo Dropout and ensemble variance estimation.
* **[P5] Real-Time Edge Deployment:** Export to ONNX Runtime/C++ for sub-second MWD streaming inference.
## 🔬 Gaps Addressed by This Project
1. Elimination of Manual Log Correlation Latency: Replaces subjective, human-in-the-loop curve matching with reproducible Dynamic Time Warping and Transformer-driven feature inference.

2. Unified Continuous Space Tabular Embedding: Solves the tabular feature-interaction bottleneck in geosciences by applying self-attention across physical drilling coordinates and petrophysical logs simultaneously.

3. Outlier-Resilient Stratigraphic Scaling: Demonstrates that pairing RobustScaler with continuous sequence transformers prevents extreme natural radioactivity spikes from destabilizing wellbore depth tracking.

## 🤝 Contributing & License
Contributions are welcome. Please open an issue or submit a pull request for improvements in feature extraction, stratigraphic modeling, or model architectures.Distributed under the MIT License. See LICENSE for more information.
