# Performance Report: Experiment 11: Decoupled S^1 / SO(2) [Sin, Cos] Embedding + Full Kinematics
**Category:** Decoupled Dedicated | **Identifier:** `Exp11_Decoupled_SinCos`

---

## 1. Architectural Configuration
* **Description:** Decoupled models predicting 2D continuous vector [sin(theta), cos(theta)] with full 9-axis inputs.
* **Changes Made:** Target bounded strictly in [-1, +1] on S^1 manifold. Eliminates multi-spin unbounded drift.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 1.86° | 2.07° | 7.45° | **3.79°** |
| **D2** | 2.26° | 2.41° | 4.29° | **2.99°** |
| **D3** | 3.58° | 2.01° | 10.28° | **5.29°** |
| **D4** | 24.13° | 2.66° | 23.39° | **16.73°** |
| **D5** | 11.91° | 3.74° | 18.84° | **11.50°** |
| **D6** | 49.30° | 13.05° | 49.42° | **37.26°** |
| **OVERALL MEAN** | **15.51°** | **4.32°** | **18.94°** | **12.93°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp11_Decoupled_SinCos_SO2.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
