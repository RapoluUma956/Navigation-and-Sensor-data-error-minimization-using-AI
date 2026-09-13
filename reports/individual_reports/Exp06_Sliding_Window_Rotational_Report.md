# Experiment 6: Rotational Kinematic Training (Sliding Window)

## 1. Overview & Objective
Train the network on continuous overlapping kinematic transitions to directly learn the angular velocity integration differential d(theta)/dt.

* **Architecture**: `LSTM(64) -> Dropout(0.3) -> LSTM(32) -> Dense(3)`
* **Hyperparameters**: Window=20, Stride_Train=5, Stride_Test=1, Epochs=60, Batch=128, Adam(lr=0.001)
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* Overlapping sliding window generator (stride=5 for training, stride=1 for testing)
* Window size expanded to 20 samples (200ms window) matching physical vehicle natural frequencies
* Drastically cut Roll error on D5 from 1.2427 rad to 0.2232 rad (82.0% error reduction!)

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0504 rad (2.89°) | 0.0432 rad (2.48°) | 0.1430 rad (8.19°) | **0.0789 rad (4.52°)** |
| **D2.xlsx** | 0.0613 rad (3.51°) | 0.0593 rad (3.40°) | 0.1274 rad (7.30°) | **0.0827 rad (4.74°)** |
| **D3.xlsx** | 0.0594 rad (3.40°) | 0.0339 rad (1.94°) | 0.0497 rad (2.85°) | **0.0477 rad (2.73°)** |
| **D4.xlsx** | 0.2856 rad (16.37°) | 0.0554 rad (3.18°) | 0.3721 rad (21.32°) | **0.2377 rad (13.62°)** |
| **D5.xlsx** | 0.2232 rad (12.79°) | 0.0534 rad (3.06°) | 0.4321 rad (24.76°) | **0.2362 rad (13.53°)** |
| **D6.xlsx** | 0.9366 rad (53.66°) | 0.2869 rad (16.44°) | 1.1222 rad (64.29°) | **0.7819 rad (44.80°)** |
| **Mean Across All Datasets** | **0.2694 rad (15.44°)** | **0.0887 rad (5.08°)** | **0.3744 rad (21.45°)** | **`0.2442 rad (13.99°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Rotational Kinematic Training (Sliding Window)`.
* **Colab Execution**: Open `Exp6_Sliding_Window_Rotational.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
