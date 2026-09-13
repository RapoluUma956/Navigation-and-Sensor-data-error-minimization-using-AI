# Performance Report: Experiment 07: Temporal Self-Attention Mechanism
**Category:** Combined 3-Axis | **Identifier:** `Exp07_Combined_Best_Practices`

---

## 1. Architectural Configuration
* **Description:** Multi-head scaled dot-product attention placed over recurrent LSTM hidden state sequences.
* **Changes Made:** Attention dynamically weights history steps based on maneuver relevance, avoiding reliance on final step.

---

## 2. Benchmark Metrics Summary Table

| Dataset | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 3D Avg RMSE (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **D1** | 2.89° | 3.39° | 9.40° | **5.23°** |
| **D2** | 2.28° | 2.70° | 10.20° | **5.06°** |
| **D3** | 4.04° | 4.73° | 11.60° | **6.79°** |
| **D4** | 6.47° | 7.67° | 33.46° | **15.87°** |
| **D5** | 69.61° | 8.47° | 35.41° | **37.83°** |
| **D6** | 6.25° | 6.89° | 21.72° | **11.61°** |
| **OVERALL MEAN** | **15.26°** | **5.64°** | **20.30°** | **13.73°** |

---

## 3. Included Artifacts in this Directory:
* **Notebook:** `Exp07_Temporal_Attention.ipynb` (Colab & Jupyter compatible)
* **Dataset Tracking Plots:** `tracking_D1.png` to `tracking_D6.png`
* **Performance Overview:** `summary_barchart.png`
* **Raw Metrics:** `results_summary.csv`
