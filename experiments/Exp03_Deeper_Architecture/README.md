# Experiment 3: Hierarchical Deeper Recurrent Architecture

## 1. Overview & Objective
Assess whether deep 3-layer hierarchical recurrent feature abstraction improves non-linear sensor fusion compared to shallow 2-layer networks.

* **Architecture**: `LSTM(128) -> Dropout(0.3) -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(3)`
* **Hyperparameters**: Timestep=10, Epochs=60, Batch=128, Optimizer=Adam(lr=0.001), ReduceLROnPlateau
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* 3 stacked recurrent layers with progressive dimensionality reduction (128 -> 64 -> 32)
* Layer-specific progressive dropout (0.3 -> 0.2 -> 0.0) to prevent co-adaptation
* Significantly improved Roll estimation on high-dynamic datasets

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0718 rad (4.11°) | 0.0814 rad (4.66°) | 0.1831 rad (10.49°) | **0.1121 rad (6.42°)** |
| **D2.xlsx** | 0.0625 rad (3.58°) | 0.0859 rad (4.92°) | 0.1829 rad (10.48°) | **0.1105 rad (6.33°)** |
| **D3.xlsx** | 0.0637 rad (3.65°) | 0.0597 rad (3.42°) | 0.0979 rad (5.61°) | **0.0738 rad (4.23°)** |
| **D4.xlsx** | 0.2446 rad (14.01°) | 0.0394 rad (2.26°) | 0.3654 rad (20.94°) | **0.2165 rad (12.40°)** |
| **D5.xlsx** | 0.3632 rad (20.81°) | 0.0797 rad (4.57°) | 0.4445 rad (25.47°) | **0.2958 rad (16.95°)** |
| **D6.xlsx** | 0.8699 rad (49.84°) | 0.2641 rad (15.13°) | 1.0830 rad (62.05°) | **0.7390 rad (42.34°)** |
| **Mean Across All Datasets** | **0.2793 rad (16.00°)** | **0.1017 rad (5.83°)** | **0.3928 rad (22.51°)** | **`0.2579 rad (14.78°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Hierarchical Deeper Recurrent Architecture`.
* **Colab Execution**: Open `Exp3_Deeper_Architecture.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
