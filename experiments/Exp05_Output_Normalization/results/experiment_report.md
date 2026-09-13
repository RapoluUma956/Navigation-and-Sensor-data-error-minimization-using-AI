# Performance Report: Experiment 05: Bidirectional LSTM (Bi-LSTM)
**Category:** Combined 3-Axis | **Identifier:** `Exp05_Output_Normalization`

---

## 1. Architectural Configuration
* **Description:** Bidirectional temporal processing concatenating forward and backward hidden representations.
* **Changes Made:** Forward and backward hidden states concatenated (128 features), integrating future and past context.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 3.03° | 3.54° | 9.75° | **5.44°** |
| **D2** | 2.39° | 2.82° | 10.57° | **5.26°** |
| **D3** | 4.21° | 4.92° | 12.06° | **7.06°** |
| **D4** | 6.77° | 7.99° | 34.52° | **16.43°** |
| **D5** | 72.08° | 8.82° | 36.38° | **39.09°** |
| **D6** | 6.53° | 7.19° | 22.49° | **12.07°** |
| **OVERALL MEAN** | **15.84°** | **5.88°** | **20.96°** | **14.22°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp05_Bidirectional_LSTM.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
