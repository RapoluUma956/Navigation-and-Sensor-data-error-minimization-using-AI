# Performance Report: Experiment 08: Stacked Deep Bi-LSTM with Dense Refinement (Best Combined)
**Category:** Combined 3-Axis | **Identifier:** `Exp08_Stacked_BiLSTM_Dense_Best`

---

## 1. Architectural Configuration
* **Description:** Deep multi-layer Bi-LSTM (64+32 units) + Dense(32, LeakyReLU) + Dense(3). Champion combined model.
* **Changes Made:** Hierarchical multi-scale temporal extraction. Slashed combined mean error to 0.2236 rad (12.81 deg).

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 2.68° | 3.16° | 8.75° | **4.86°** |
| **D2** | 2.12° | 2.51° | 9.48° | **4.70°** |
| **D3** | 3.75° | 4.40° | 10.80° | **6.32°** |
| **D4** | 6.03° | 7.13° | 31.23° | **14.79°** |
| **D5** | 65.03° | 7.88° | 33.12° | **35.34°** |
| **D6** | 5.82° | 6.42° | 20.28° | **10.84°** |
| **OVERALL MEAN** | **14.24°** | **5.25°** | **18.94°** | **12.81°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp08_Stacked_Deep_BiLSTM.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
