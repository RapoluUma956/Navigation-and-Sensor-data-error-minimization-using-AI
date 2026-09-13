# Performance Report: Experiment 10.b: Direct Gyro Kinematic Skip Connection + Dynamic Windowing
**Category:** Optimization Variant | **Identifier:** `Exp10_Optimizations_10b`

---

## 1. Architectural Configuration
* **Description:** Direct linear bypass from instantaneous gyro rate (p, q, r) to output + dynamic W (15/25).
* **Changes Made:** Bypasses recurrence for rate tracking. Winner on steep climb D4 (13.46 deg) and D2 Yaw (3.51 deg).

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 2.00° | 6.16° | 6.65° | **4.94°** |
| **D2** | 1.62° | 2.09° | 3.51° | **2.41°** |
| **D3** | 2.77° | 1.98° | 7.32° | **4.02°** |
| **D4** | 19.47° | 1.98° | 18.91° | **13.45°** |
| **D5** | 10.37° | 2.69° | 15.35° | **9.47°** |
| **D6** | 103.08° | 9.85° | 100.20° | **71.05°** |
| **OVERALL MEAN** | **23.22°** | **4.12°** | **25.32°** | **17.56°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp10b_Gyro_Skip_Dynamic_Window.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
