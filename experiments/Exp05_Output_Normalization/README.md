# Experiment 5: Dual Standardization (Input & Target Normalization)

## 1. Overview & Objective
Resolve target variance imbalance where high-variance Roll swings dominate Mean Squared Error loss and starve low-variance Pitch and Yaw updates.

* **Architecture**: `LSTM(64) -> Dropout(0.25) -> LSTM(32) -> Dense(3)`
* **Hyperparameters**: Timestep=10, Epochs=60, Batch=128, StandardScaler for X and Y, Adam(lr=0.001)
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* Applied independent StandardScaler to both 9-DOF input features and 3-axis Euler angle targets
* Equalized gradient magnitude across Roll, Pitch, and Yaw regardless of physical operating variance
* Achieved lowest error on the extreme D6 tumbling dataset (0.7131 rad) and 2nd place overall (0.2356 rad)

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0711 rad (4.07°) | 0.0552 rad (3.16°) | 0.0954 rad (5.47°) | **0.0739 rad (4.23°)** |
| **D2.xlsx** | 0.0634 rad (3.63°) | 0.0423 rad (2.42°) | 0.0778 rad (4.46°) | **0.0612 rad (3.50°)** |
| **D3.xlsx** | 0.0778 rad (4.46°) | 0.0430 rad (2.47°) | 0.0934 rad (5.35°) | **0.0714 rad (4.09°)** |
| **D4.xlsx** | 0.2368 rad (13.57°) | 0.0398 rad (2.28°) | 0.2809 rad (16.10°) | **0.1859 rad (10.65°)** |
| **D5.xlsx** | 0.5753 rad (32.96°) | 0.0380 rad (2.18°) | 0.3114 rad (17.84°) | **0.3082 rad (17.66°)** |
| **D6.xlsx** | 0.7706 rad (44.15°) | 0.2181 rad (12.49°) | 1.1505 rad (65.92°) | **0.7131 rad (40.86°)** |
| **Mean Across All Datasets** | **0.2992 rad (17.14°)** | **0.0727 rad (4.17°)** | **0.3349 rad (19.19°)** | **`0.2356 rad (13.50°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Dual Standardization (Input & Target Normalization)`.
* **Colab Execution**: Open `Exp5_Output_Normalization.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
