# Performance Report: Experiment 10: Decoupled Single-Axis + Continuous Phase Unwrapping (PROJECT CHAMPION)
**Category:** Decoupled Dedicated | **Identifier:** `Exp10_Continuous_Phase_Unwrapping`

---

## 1. Architectural Configuration
* **Description:** Decoupled sub-networks + continuous phase unwrapping (np.unwrap) + geodesic evaluation.
* **Changes Made:** Eliminated +-pi boundary jump discontinuities. D5 roll error slashed to 9.44 deg. 4.73 deg overall mean.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 1.36° | 2.15° | 4.99° | **2.84°** |
| **D2** | 1.17° | 1.82° | 4.07° | **2.35°** |
| **D3** | 2.28° | 2.84° | 6.37° | **3.83°** |
| **D4** | 2.76° | 3.52° | 8.14° | **4.81°** |
| **D5** | 9.44° | 4.27° | 10.20° | **7.97°** |
| **D6** | 3.41° | 4.08° | 12.26° | **6.58°** |
| **OVERALL MEAN** | **3.41°** | **3.11°** | **7.67°** | **4.73°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp10_Decoupled_Phase_Unwrapping_Winner.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
