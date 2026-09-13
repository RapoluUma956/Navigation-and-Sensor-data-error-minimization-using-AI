# Performance Report: Experiment 06: Hybrid 1D-CNN + LSTM Architecture
**Category:** Combined 3-Axis | **Identifier:** `Exp06_Sliding_Window_Rotational`

---

## 1. Architectural Configuration
* **Description:** 1D Convolutional front-end (32 filters, k=3, ReLU) followed by LSTM recurrent layer.
* **Changes Made:** Added spatial convolution to adaptively filter high-frequency sensor noise and rotor vibration.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 3.19° | 3.70° | 10.17° | **5.68°** |
| **D2** | 2.51° | 2.97° | 11.00° | **5.49°** |
| **D3** | 4.42° | 5.16° | 12.55° | **7.38°** |
| **D4** | 7.08° | 8.37° | 35.98° | **17.14°** |
| **D5** | 75.17° | 9.25° | 37.70° | **40.71°** |
| **D6** | 6.83° | 7.51° | 23.45° | **12.59°** |
| **OVERALL MEAN** | **16.53°** | **6.16°** | **21.81°** | **14.83°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp06_Hybrid_CNN_LSTM.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
