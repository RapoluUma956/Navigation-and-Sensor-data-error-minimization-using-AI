# Performance Report: Experiment 09: Decoupled Dedicated Single-Axis Architecture (Raw Angles)
**Category:** Decoupled Dedicated | **Identifier:** `Exp09_Decoupled_Single_Axis`

---

## 1. Architectural Configuration
* **Description:** 3 independent sub-networks for Roll, Pitch, and Yaw with isolated parameters and loss gradients.
* **Changes Made:** Eliminated multi-axis gradient interference. D2 cruising error slashed to 2.39 deg.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 1.39° | 2.18° | 5.06° | **2.88°** |
| **D2** | 1.20° | 1.84° | 4.14° | **2.39°** |
| **D3** | 2.36° | 2.92° | 6.53° | **3.94°** |
| **D4** | 4.24° | 4.67° | 23.61° | **10.84°** |
| **D5** | 67.72° | 5.10° | 27.79° | **33.54°** |
| **D6** | 3.55° | 4.24° | 12.66° | **6.82°** |
| **OVERALL MEAN** | **13.41°** | **3.49°** | **13.30°** | **10.07°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp09_Decoupled_Raw_Angles.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
