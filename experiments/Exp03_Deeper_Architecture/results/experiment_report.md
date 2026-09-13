# Performance Report: Experiment 03: Temporal Window Size Optimization (W=20)
**Category:** Combined 3-Axis | **Identifier:** `Exp03_Deeper_Architecture`

---

## 1. Architectural Configuration
* **Description:** Expanded temporal sequence length from W=10 to W=20 for longer integration receptive field.
* **Changes Made:** Doubled sequence length to W=20, allowing model to capture lower-frequency gyro bias walk.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 3.37° | 3.92° | 10.78° | **6.03°** |
| **D2** | 2.66° | 3.15° | 11.69° | **5.83°** |
| **D3** | 4.69° | 5.47° | 13.28° | **7.81°** |
| **D4** | 7.53° | 8.89° | 37.94° | **18.12°** |
| **D5** | 80.10° | 9.81° | 39.99° | **43.30°** |
| **D6** | 7.27° | 7.99° | 24.75° | **13.34°** |
| **OVERALL MEAN** | **17.60°** | **6.54°** | **23.07°** | **15.74°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp03_Window_Size_W20.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
