# Performance Report: Experiment 01: Baseline LSTM Architecture
**Category:** Combined 3-Axis | **Identifier:** `Exp01_Baseline_Fixed`

---

## 1. Architectural Configuration
* **Description:** Vanilla single-layer LSTM (64 units, W=10, Dropout=0.2, Dense(3)). Standard baseline reference.
* **Changes Made:** Baseline architecture implementation. No feature engineering or decoupling.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 3.74° | 4.40° | 11.54° | **6.56°** |
| **D2** | 2.99° | 3.52° | 12.52° | **6.34°** |
| **D3** | 5.12° | 5.97° | 14.10° | **8.40°** |
| **D4** | 8.14° | 9.65° | 39.60° | **19.13°** |
| **D5** | 83.54° | 10.55° | 41.51° | **45.20°** |
| **D6** | 7.93° | 8.66° | 25.90° | **14.16°** |
| **OVERALL MEAN** | **18.58°** | **7.13°** | **24.20°** | **16.63°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp01_Baseline_Fixed.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
