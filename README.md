# 🚦 ML-Based Traffic Congestion & Incident Detection Using Aerial Imagery


An end-to-end supervised learning pipeline that detects traffic congestion and road incidents using features derived from aerial camera feeds. The project performs comprehensive Exploratory Data Analysis (EDA), trains and compares six classical ML models, and selects the best-performing classifier through multi-objective (Pareto) analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [1 — Data Understanding & EDA](#1--data-understanding--eda)
  - [2 — Model Training & Comparison](#2--model-training--comparison)
  - [3 — Best-Model Deep-Dive](#3--best-model-deep-dive)
- [Results](#results)
- [Visualisations](#visualisations)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

The objective of this project is to design an AI pipeline capable of classifying road segments as **Normal (0)** or **Congestion / Incident (1)** based on 10 features extracted from aerial imagery tiles.

The project compares multiple machine learning algorithms, evaluates them on standard classification metrics, and performs Pareto-optimal analysis across **Precision**, **Recall**, and **Inference Latency** to recommend the best model for real-world deployment.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | `Traffic Congestion - Sheet1.csv` |
| **Samples** | 4,000 |
| **Features** | 10 (all `float64`) |
| **Target** | `label` — binary (`0` = Normal, `1` = Congestion/Incident) |
| **Class Balance** | ~55 % Normal / ~45 % Congestion |
| **Missing Values** | 0 |
| **Duplicate Rows** | 0 |

---

## Features

| Feature | Description | Relevance |
|---|---|---|
| `vehicle_density` | Vehicles per unit area in the tile | **HIGH** — directly indicates congestion |
| `avg_vehicle_speed` | Mean speed of detected vehicles | **HIGH** — lower speeds → congestion |
| `speed_std` | Std. deviation of vehicle speeds | **MEDIUM** — stop-and-go patterns |
| `lane_occupancy` | Fraction of lane space occupied (0–1) | **HIGH** — less free space = congestion |
| `queue_length` | Estimated vehicle queue length | **HIGH** — longer queues = backup |
| `edge_density` | Density of edges in imagery | **MEDIUM** — more vehicles/objects |
| `optical_flow_mag` | Magnitude of optical flow (motion) | **HIGH** — lower motion = stopped traffic |
| `shadow_fraction` | Proportion of image with shadows | **LOW** — environmental noise |
| `time_of_day_norm` | Normalised time of day (0–1) | **LOW** — contextual feature |
| `road_width_norm` | Normalised road width | **LOW** — static infrastructure feature |

### Key EDA Insights

- **Strong indicators:** `avg_vehicle_speed` (r = −0.69), `lane_occupancy` (r = +0.67), `vehicle_density` (r = +0.67), `optical_flow_mag` (r = −0.63), `edge_density` (r = +0.61), `queue_length` (r = +0.60).
- **Weak / noise features:** `shadow_fraction`, `time_of_day_norm`, `road_width_norm` — near-zero correlation with the target.
- **Multicollinearity detected:**
  - `vehicle_density` ↔ `lane_occupancy` (r = 0.97)
  - `avg_vehicle_speed` ↔ `optical_flow_mag` (r = 0.90)

---

## Project Structure

```
Traffic-congestion-incident-detection/
├── Traffic_Congestion_Capstone_Executed.ipynb   # Main Jupyter notebook (fully executed)
├── README.md                                    # This file
├── EDA/                                         # EDA visualisation exports
│   ├── EDA_Visualizations.png
│   ├── correlation_matrix.png
│   ├── feature_distributions.png
│   ├── risk_map.png
│   ├── speed_queue_heatmaps.png
│   ├── target_distribution.png
│   └── traffic_heatmaps.png
└── BEST_MODEL_ANALYSIS/                         # Best model evaluation plots
    ├── action_priority.png
    ├── best_model_evaluation.png
    ├── model_comparison.png
    ├── pareto_2d_tradeoffs.png
    └── pareto_3d_frontier.png
```

---

## Pipeline

### 1 — Data Understanding & EDA

- Load & inspect the dataset (shape, types, missing values, duplicates).
- Compute descriptive statistics and class-wise feature distributions.
- Visualise:
  - Feature distributions & box-plots
  - Correlation matrix (Pearson)
  - Speed / queue-length heatmaps
  - Target distribution (class balance)
  - Spatial congestion risk maps

### 2 — Model Training & Comparison

**Data Preparation**
- 80 / 20 stratified train-test split (`random_state=42`).
- Feature scaling with `StandardScaler` (applied to distance-based models).

**Models Compared**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.8475 | 0.8459 | 0.8083 | 0.8267 | 0.9300 |
| **SVM** | 0.8475 | 0.8480 | 0.8056 | 0.8262 | 0.9210 |
| **Random Forest** | 0.8438 | 0.8507 | 0.7917 | 0.8201 | 0.9253 |
| **Gradient Boosting** | 0.8375 | 0.8382 | 0.7917 | 0.8143 | 0.9229 |
| **KNN** | 0.8113 | 0.7817 | 0.8056 | 0.7934 | 0.8931 |
| **Decision Tree** | 0.7763 | 0.7388 | 0.7778 | 0.7578 | 0.7764 |

### 3 — Best-Model Deep-Dive

- **Pareto-optimal analysis** over Precision, Recall, and Inference Latency (2-D trade-off plots + 3-D frontier).
- Detailed evaluation of the selected best model including:
  - Confusion matrix
  - Classification report
  - ROC curve
  - Precision-Recall curve
  - Action priority analysis for congestion management

---

## Results

- **Logistic Regression** emerges as the top performer on the primary metric (ROC-AUC = 0.930) and achieves the best F1-Score (0.827), balancing precision and recall for congestion/incident detection.
- **SVM** performs comparably (ROC-AUC = 0.921) but with slightly higher latency.
- **Decision Tree** is the weakest model, indicating that the decision boundary in this feature space is not well captured by axis-aligned splits.
- Pareto analysis confirms that simpler models (Logistic Regression) can dominate more complex ones when latency is also considered.

---

## Visualisations

All saved visualisation artefacts are located in two directories:

### `EDA/`
| File | Description |
|---|---|
| `EDA_Visualizations.png` | Combined EDA dashboard |
| `correlation_matrix.png` | Pearson correlation heatmap |
| `feature_distributions.png` | Histograms for each feature |
| `risk_map.png` | Spatial congestion risk map |
| `speed_queue_heatmaps.png` | Speed & queue-length heatmaps |
| `target_distribution.png` | Class distribution bar chart |
| `traffic_heatmaps.png` | Actual vs predicted congestion heatmaps |

### `BEST_MODEL_ANALYSIS/`
| File | Description |
|---|---|
| `model_comparison.png` | Side-by-side metric comparison across all models |
| `best_model_evaluation.png` | Confusion matrix, ROC, PR curves for the best model |
| `pareto_2d_tradeoffs.png` | 2-D Pareto trade-off plots |
| `pareto_3d_frontier.png` | 3-D Pareto frontier visualisation |
| `action_priority.png` | Congestion action-priority analysis |

---

## Getting Started

### Prerequisites

- Python ≥ 3.9
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Traffic-congestion-incident-detection.git
cd Traffic-congestion-incident-detection

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt   # see Dependencies below
```

### Run

```bash
jupyter notebook Traffic_Congestion_Capstone_Executed.ipynb
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` | Plotting / visualisation |
| `seaborn` | Statistical visualisation |
| `scikit-learn` | ML models, metrics, preprocessing |

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## License

This project is for academic purposes. Feel free to use and adapt with attribution.
