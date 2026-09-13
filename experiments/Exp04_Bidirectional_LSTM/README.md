# Experiment 4: Bidirectional Sequential Modeling

## 1. Overview & Objective
Harness forward and reverse sequential context within temporal sliding windows to reconstruct attitude trajectories and eliminate phase lag.

* **Architecture**: `Bidirectional(LSTM(64)) -> Dropout(0.3) -> LSTM(32) -> Dense(3)`
* **Hyperparameters**: Timestep=10, Epochs=60, Batch=128, Optimizer=Adam(lr=0.001)
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* Bidirectional LSTM layer processes IMU input sequence in both forward and backward temporal directions
* Forward pass estimates current attitude from past acceleration, backward pass correlates deceleration profile
* Achieved 1st place on Dataset D3 with 0.0405 rad RMSE

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0702 rad (4.02°) | 0.0524 rad (3.00°) | 0.1711 rad (9.80°) | **0.0979 rad (5.61°)** |
| **D2.xlsx** | 0.0892 rad (5.11°) | 0.0567 rad (3.25°) | 0.1430 rad (8.20°) | **0.0963 rad (5.52°)** |
| **D3.xlsx** | 0.0390 rad (2.24°) | 0.0267 rad (1.53°) | 0.0556 rad (3.19°) | **0.0405 rad (2.32°)** |
| **D4.xlsx** | 0.2840 rad (16.27°) | 0.0427 rad (2.45°) | 0.3028 rad (17.35°) | **0.2098 rad (12.02°)** |
| **D5.xlsx** | 0.4138 rad (23.71°) | 0.0629 rad (3.60°) | 0.4453 rad (25.51°) | **0.3074 rad (17.61°)** |
| **D6.xlsx** | 0.9898 rad (56.71°) | 0.1898 rad (10.88°) | 1.0240 rad (58.67°) | **0.7345 rad (42.09°)** |
| **Mean Across All Datasets** | **0.3143 rad (18.01°)** | **0.0719 rad (4.12°)** | **0.3570 rad (20.45°)** | **`0.2477 rad (14.19°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Bidirectional Sequential Modeling`.
* **Colab Execution**: Open `Exp4_Bidirectional_LSTM.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
