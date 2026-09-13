# Performance Report: Experiment 10.c: 72-Config Grid Sweep & Optimal P* (W=16, Huber)
**Category:** Optimization Variant | **Identifier:** `Exp10_Optimizations_10c`

---

## 1. Architectural Configuration
* **Description:** Systematic grid sweep over W in {12, 16, 20, 24} and losses {mse, huber, log_cosh}.
* **Changes Made:** Identified optimal configuration P* (W=16, Huber loss). D6 roll error dropped to 88.57 deg.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 2.27° | 1.74° | 7.53° | **3.85°** |
| **D2** | 1.71° | 1.66° | 4.42° | **2.60°** |
| **D3** | 2.31° | 2.81° | 10.70° | **5.27°** |
| **D4** | 24.42° | 2.60° | 23.02° | **16.68°** |
| **D5** | 10.82° | 3.99° | 16.17° | **10.32°** |
| **D6** | 88.57° | 11.17° | 102.60° | **67.45°** |
| **OVERALL MEAN** | **21.68°** | **3.99°** | **27.41°** | **17.70°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp10c_Hyperparameter_Grid_Optimal.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
