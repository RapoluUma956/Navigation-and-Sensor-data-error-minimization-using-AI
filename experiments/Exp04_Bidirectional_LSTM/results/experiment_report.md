# Performance Report: Experiment 04: Gated Recurrent Unit (GRU) Benchmark
**Category:** Combined 3-Axis | **Identifier:** `Exp04_Bidirectional_LSTM`

---

## 1. Architectural Configuration
* **Description:** Replaced LSTM cell with Gated Recurrent Unit (64 units) without internal cell state.
* **Changes Made:** Eliminated cell memory c_t; reset and update gates cut parameter count by 25% for faster inference.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 3.66° | 4.27° | 11.36° | **6.43°** |
| **D2** | 2.89° | 3.43° | 12.26° | **6.19°** |
| **D3** | 5.00° | 5.82° | 13.89° | **8.23°** |
| **D4** | 7.95° | 9.41° | 39.08° | **18.81°** |
| **D5** | 82.22° | 10.34° | 41.02° | **44.53°** |
| **D6** | 7.71° | 8.47° | 25.58° | **13.92°** |
| **OVERALL MEAN** | **18.24°** | **6.95°** | **23.87°** | **16.35°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp04_GRU_Benchmark.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
