# Experiment 2: Hyperparameter Optimization & Adaptive Scheduling

## 1. Overview & Objective
Evaluate the impact of expanding temporal receptive field from 2 to 10 samples and introducing modern adaptive optimization with learning rate plateau decay.

* **Architecture**: `LSTM(50) -> Dropout(0.25) -> LSTM(20) -> Dense(3)`
* **Hyperparameters**: Timestep=10, Epochs=50, Batch=128, Optimizer=Adam(lr=0.001), ReduceLROnPlateau(patience=5, factor=0.5)
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* Increased sequence length from 2 to 10 timesteps (100ms receptive field at 100Hz)
* Switched from RMSprop to Adam with adaptive learning rate reduction on plateau
* Increased batch size to 128 for smoother mini-batch gradient updates

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0801 rad (4.59°) | 0.0653 rad (3.74°) | 0.1721 rad (9.86°) | **0.1058 rad (6.06°)** |
| **D2.xlsx** | 0.0791 rad (4.53°) | 0.0722 rad (4.14°) | 0.1991 rad (11.41°) | **0.1168 rad (6.69°)** |
| **D3.xlsx** | 0.0989 rad (5.67°) | 0.0704 rad (4.03°) | 0.1164 rad (6.67°) | **0.0953 rad (5.46°)** |
| **D4.xlsx** | 0.3237 rad (18.54°) | 0.0395 rad (2.26°) | 0.3242 rad (18.57°) | **0.2291 rad (13.13°)** |
| **D5.xlsx** | 0.5443 rad (31.18°) | 0.0650 rad (3.72°) | 0.4593 rad (26.32°) | **0.3562 rad (20.41°)** |
| **D6.xlsx** | 0.9585 rad (54.92°) | 0.2154 rad (12.34°) | 1.1716 rad (67.13°) | **0.7819 rad (44.80°)** |
| **Mean Across All Datasets** | **0.3474 rad (19.91°)** | **0.0880 rad (5.04°)** | **0.4071 rad (23.33°)** | **`0.2808 rad (16.09°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Hyperparameter Optimization & Adaptive Scheduling`.
* **Colab Execution**: Open `Exp2_Hyperparameter_Tuning.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
