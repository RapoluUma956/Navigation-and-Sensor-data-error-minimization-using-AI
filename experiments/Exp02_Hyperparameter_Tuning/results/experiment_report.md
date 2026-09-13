# Performance Report: Experiment 02: Feature Engineering & Derived Kinematics
**Category:** Combined 3-Axis | **Identifier:** `Exp02_Hyperparameter_Tuning`

---

## 1. Architectural Configuration
* **Description:** 12-channel input incorporating acceleration norm |a|, gyro rate norm |w|, and finite deltas.
* **Changes Made:** Expanded input dimensionality from 6 to 12 channels. Network kept identical to baseline.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 3.50° | 4.08° | 11.14° | **6.24°** |
| **D2** | 2.80° | 3.28° | 12.06° | **6.04°** |
| **D3** | 4.82° | 5.66° | 13.70° | **8.06°** |
| **D4** | 7.75° | 9.17° | 38.67° | **18.53°** |
| **D5** | 81.42° | 10.11° | 40.63° | **44.05°** |
| **D6** | 7.51° | 8.24° | 25.27° | **13.67°** |
| **OVERALL MEAN** | **17.97°** | **6.76°** | **23.58°** | **16.10°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp02_Feature_Engineering.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
