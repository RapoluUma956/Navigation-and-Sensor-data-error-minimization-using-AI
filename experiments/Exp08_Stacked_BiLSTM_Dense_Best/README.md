# Experiment 8: Stacked Bi-LSTM with Dense Feature Projection (Winning Combined Model)

## 1. Overview & Objective
Maximize feature capacity with bidirectional recurrence across all recurrent tiers, paired with an intermediate nonlinear Dense projection layer.

* **Architecture**: `Bidirectional(LSTM(128)) -> Dropout(0.3) -> Bidirectional(LSTM(64)) -> Dense(64, relu) -> Dropout(0.2) -> Dense(3)`
* **Hyperparameters**: Window=20, Stride=5/1, Dual StandardScaler, Cosine Decay / Plateau Scheduler, Epochs=80
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* Two fully bidirectional LSTM layers capturing hierarchical temporal representations
* Intermediate Dense(64, activation=relu) projection layer maps temporal features to attitude manifolds
* OVERALL BENCHMARK WINNER: 0.2236 rad average 3D attitude RMSE across all 6 datasets (23.0% error reduction vs baseline)

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0487 rad (2.79°) | 0.0345 rad (1.98°) | 0.0972 rad (5.57°) | **0.0602 rad (3.45°)** |
| **D2.xlsx** | 0.0258 rad (1.48°) | 0.0224 rad (1.28°) | 0.0476 rad (2.73°) | **0.0319 rad (1.83°)** |
| **D3.xlsx** | 0.0422 rad (2.42°) | 0.0255 rad (1.46°) | 0.0858 rad (4.92°) | **0.0512 rad (2.93°)** |
| **D4.xlsx** | 0.3447 rad (19.75°) | 0.0335 rad (1.92°) | 0.3252 rad (18.63°) | **0.2345 rad (13.43°)** |
| **D5.xlsx** | 0.4073 rad (23.34°) | 0.0463 rad (2.65°) | 0.2436 rad (13.96°) | **0.2324 rad (13.32°)** |
| **D6.xlsx** | 0.9126 rad (52.29°) | 0.2154 rad (12.34°) | 1.0658 rad (61.07°) | **0.7313 rad (41.90°)** |
| **Mean Across All Datasets** | **0.2969 rad (17.01°)** | **0.0629 rad (3.60°)** | **0.3109 rad (17.81°)** | **`0.2236 rad (12.81°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Stacked Bi-LSTM with Dense Feature Projection (Winning Combined Model)`.
* **Colab Execution**: Open `Exp8_Stacked_BiLSTM_Dense_Best.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
