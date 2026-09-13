# Experiment 7: Multi-Technique Combined Architecture

## 1. Overview & Objective
Synthesize all winning experimental features into a high-capacity unified recurrent network with early stopping regularisation.

* **Architecture**: `Bidirectional(LSTM(128)) -> Dropout(0.3) -> LSTM(64) -> Dense(3)`
* **Hyperparameters**: Sliding Window=20 (stride=5), Dual Normalization (StandardScaler), Epochs=100, EarlyStopping(patience=10)
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* Merged Bidirectional LSTM front-end, dual StandardScaler, and overlapping sliding window generator
* Added EarlyStopping with patience=10 and restore_best_weights to prevent overfitting
* Strong performance on low and medium dynamics datasets (D1: 0.0606 rad, D2: 0.0383 rad)

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0372 rad (2.13°) | 0.0313 rad (1.79°) | 0.1133 rad (6.49°) | **0.0606 rad (3.47°)** |
| **D2.xlsx** | 0.0389 rad (2.23°) | 0.0234 rad (1.34°) | 0.0525 rad (3.01°) | **0.0383 rad (2.19°)** |
| **D3.xlsx** | 0.0446 rad (2.55°) | 0.0364 rad (2.08°) | 0.1032 rad (5.92°) | **0.0614 rad (3.52°)** |
| **D4.xlsx** | 0.3465 rad (19.85°) | 0.0322 rad (1.84°) | 0.3401 rad (19.49°) | **0.2396 rad (13.73°)** |
| **D5.xlsx** | 0.6787 rad (38.88°) | 0.0597 rad (3.42°) | 0.2710 rad (15.53°) | **0.3365 rad (19.28°)** |
| **D6.xlsx** | 1.0421 rad (59.71°) | 0.3286 rad (18.83°) | 1.1190 rad (64.11°) | **0.8299 rad (47.55°)** |
| **Mean Across All Datasets** | **0.3647 rad (20.89°)** | **0.0853 rad (4.88°)** | **0.3332 rad (19.09°)** | **`0.2610 rad (14.96°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Multi-Technique Combined Architecture`.
* **Colab Execution**: Open `Exp7_Combined_Best_Practices.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
