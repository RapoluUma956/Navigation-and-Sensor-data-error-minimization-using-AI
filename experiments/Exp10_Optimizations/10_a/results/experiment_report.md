# Performance Report: Experiment 10.a: Denser Stride (2) + Huber Loss (delta=0.1) + LeakyReLU
**Category:** Optimization Variant | **Identifier:** `Exp10_Optimizations_10a`

---

## 1. Architectural Configuration
* **Description:** Tuning Exp 10 with stride 2 (2.5x more samples), Huber loss, and LeakyReLU(0.1).
* **Changes Made:** Avoids temporal integration skips and suppresses outlier gradient shockwaves. D3 Pitch dropped to 1.21 deg.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 1.56° | 2.11° | 6.80° | **3.50°** |
| **D2** | 1.76° | 1.63° | 3.73° | **2.37°** |
| **D3** | 2.94° | 1.21° | 6.43° | **3.52°** |
| **D4** | 21.76° | 1.82° | 20.33° | **14.63°** |
| **D5** | 10.59° | 2.59° | 16.50° | **9.89°** |
| **D6** | 104.11° | 10.11° | 102.11° | **72.11°** |
| **OVERALL MEAN** | **23.79°** | **3.25°** | **25.98°** | **17.67°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp10a_Denser_Huber_LeakyReLU.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
