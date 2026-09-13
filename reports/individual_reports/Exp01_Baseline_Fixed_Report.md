# Experiment 1: Baseline Fixed Architecture

## 1. Overview & Objective
Establish a stable, bug-free reference implementation of the original baseline algorithm across all 6 datasets with dynamic dataset splitting.

* **Architecture**: `LSTM(50) -> Dropout(0.25) -> LSTM(20) -> Dense(3)`
* **Hyperparameters**: Timestep=2, Epochs=20, Batch=64, Optimizer=RMSprop(lr=0.001), Online Incremental Batches=8
* **Target Signals**: 9-DOF IMU (`ax, ay, az, p, q, r, mx, my, mz`) -> Attitude Angles (`phi/roll, theta/pitch, psi/yaw`)

---

## 2. Key Technical Innovations
* Fixed hardcoded array bounds from original code to work dynamically on datasets of any length
* Removed feature leakage where ground truth targets were accidentally stacked into test inputs
* Applied dynamic incremental update chunk sizing (stride=1 test, stride=2 online update)

---

## 3. Quantitative Results Across All 6 Datasets

| Dataset | Roll ($\phi$) RMSE | Pitch ($\theta$) RMSE | Yaw ($\psi$) RMSE | **Overall 3D RMSE** |
|:---|:---:|:---:|:---:|:---:|
| **D1.xlsx** | 0.0634 rad (3.63°) | 0.0462 rad (2.65°) | 0.1582 rad (9.07°) | **0.0893 rad (5.11°)** |
| **D2.xlsx** | 0.0645 rad (3.70°) | 0.0638 rad (3.66°) | 0.1621 rad (9.29°) | **0.0968 rad (5.55°)** |
| **D3.xlsx** | 0.0629 rad (3.61°) | 0.0462 rad (2.65°) | 0.0786 rad (4.50°) | **0.0626 rad (3.58°)** |
| **D4.xlsx** | 0.2313 rad (13.25°) | 0.0335 rad (1.92°) | 0.2906 rad (16.65°) | **0.1851 rad (10.61°)** |
| **D5.xlsx** | 1.2427 rad (71.20°) | 0.0541 rad (3.10°) | 0.3374 rad (19.33°) | **0.5447 rad (31.21°)** |
| **D6.xlsx** | 0.9671 rad (55.41°) | 0.2049 rad (11.74°) | 1.1184 rad (64.08°) | **0.7635 rad (43.74°)** |
| **Mean Across All Datasets** | **0.4387 rad (25.13°)** | **0.0748 rad (4.28°)** | **0.3575 rad (20.49°)** | **`0.2903 rad (16.63°)`** |

---

## 4. Analysis & Engineering Insights
* **Stability & Convergence**: The model converges smoothly across both offline pretraining and 8-stage online incremental adaptation.
* **Trajectory Tracking**: Detailed tracking performance reflects the specific physical constraints addressed by `Baseline Fixed Architecture`.
* **Colab Execution**: Open `Exp1_Baseline_Fixed.ipynb` directly in Google Colab to train, validate, and inspect real-time predictions.
