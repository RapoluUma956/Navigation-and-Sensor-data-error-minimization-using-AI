# Experiment 9: Kinematic Decoupled Attitude Estimation

## 1. Motivation & Problem Formulation
In a combined model predicting Roll, Pitch, and Yaw simultaneously, severe cross-axis gradient interference occurs:
1. **Variance Imbalance**: High-variance Roll swings overwhelm low-variance Pitch and Yaw, starving them of gradient power.
2. **Physical Kinematic Independence**:
   * **Roll ($\phi$)**: Governed primarily by lateral acceleration ($a_y, a_z$) resolving gravity, and roll gyro rate $p$.
   * **Pitch ($\theta$)**: Governed primarily by longitudinal acceleration ($a_x, a_z$) resolving gravity, and pitch gyro rate $q$.
   * **Yaw ($\psi$)**: Accelerometers cannot sense rotation around the gravity vector; Yaw depends strictly on the magnetometer ($m_x, m_y, m_z$) and yaw gyro rate $r$.
3. **Decoupled Architecture**: Constructing 3 dedicated, independently optimized neural networks eliminates gradient cross-talk.

```
                         ┌──► Dedicated Model 1 (Roll φ)  ──► Loss_Roll  = MSE(φ) ──► Roll  (φ)
                         │
IMU Sensor Stream ───────┼──► Dedicated Model 2 (Pitch θ) ──► Loss_Pitch = MSE(θ) ──► Pitch (θ)
(ax,ay,az,p,q,r,mx,my,mz)│
                         └──► Dedicated Model 3 (Yaw ψ)   ──► Loss_Yaw   = MSE(ψ) ──► Yaw   (ψ)
```

---

## 2. Dedicated Single-Axis Architecture
* **Input Layer**: Shape `(window_size=20, input_dim=9)`
* **Recurrent Core**: `Bidirectional(LSTM(64, return_sequences=True)) -> Dropout(0.25) -> LSTM(32)`
* **Feature Extraction**: `Dense(32, activation='relu') -> Dropout(0.15)`
* **Output**: `Dense(1, activation='linear')` dedicated purely to its specific angle.
* **Optimization**: Offline pretraining with Adam(1e-3) and ReduceLROnPlateau, followed by 8-stage online streaming adaptation with Adam(1e-4).

---

## 3. Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** | Flight Profile |
|:---|:---:|:---:|:---:|:---:|:---|
| **D1.xlsx** | **0.0279 rad (1.60°)** | **0.0411 rad (2.36°)** | 0.1190 rad (6.82°) | **0.0627 rad (3.59°)** | Standard level flight |
| **D2.xlsx** | **0.0267 rad (1.53°)** | **0.0343 rad (1.96°)** | **0.0644 rad (3.69°)** | **0.0418 rad (2.39°)** | Smooth cruising (Best) |
| **D3.xlsx** | **0.0551 rad (3.16°)** | **0.0308 rad (1.76°)** | 0.1161 rad (6.65°) | **0.0673 rad (3.86°)** | Dynamic maneuvers |
| **D4.xlsx** | 0.3873 rad (22.19°) | **0.0398 rad (2.28°)** | 0.3722 rad (21.33°) | **0.2664 rad (15.27°)** | Near-vertical pitch (79°) |
| **D5.xlsx** | **0.1648 rad (9.44°)** | **0.0554 rad (3.17°)** | 0.2527 rad (14.48°) | **0.1576 rad (9.03°)** | Boundary roll swings (±180°) |
| **D6.xlsx** | 1.8196 rad (104.26°) | **0.1815 rad (10.40°)** | 1.7712 rad (101.48°) | **1.2574 rad (72.05°)** | Continuous 360° tumbling |

*Mean 3D Attitude RMSE across all 6 datasets: **0.3089 rad (17.70°)**.*

---

## 4. Google Colab Suite
The following 4 dedicated Colab notebooks are included in this folder:
1. `Roll_Estimation_LSTM.ipynb`
2. `Pitch_Estimation_LSTM.ipynb`
3. `Yaw_Estimation_LSTM.ipynb`
4. `Decoupled_Attitude_Master_Colab.ipynb`
