# Navigation Sensor Error Minimization Using AI

## Project Title
**Navigation Sensor Error Minimization Using Long Short-Term Memory (LSTM)**

---

## Project Overview

This research project aims to minimize attitude estimation errors in navigation sensor data using Artificial Intelligence. The work focuses on developing deep learning models that learn temporal relationships from inertial sensor measurements to improve the estimation of Roll (φ), Pitch (θ), and Yaw (ψ). The project is being carried out as part of a DRDO internship, with future experiments exploring different architectures, training strategies, and optimization techniques to improve robustness and generalization.

---

## Problem Statement

Navigation systems rely on accelerometers, gyroscopes, and magnetometers to estimate the orientation of a moving platform. However, sensor noise, bias, drift, and environmental disturbances introduce errors that accumulate over time, reducing navigation accuracy. This project investigates the use of Long Short-Term Memory (LSTM) networks to learn temporal patterns in navigation sensor data and minimize attitude estimation errors while improving prediction accuracy under varying operating conditions.

---

## Project Structure

```text
Navigation-Sensor-Error-Minimization/
│
├── datasets/
│
├── experiments/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── results/
│
├── utils/
│
├── README.md
│
└── requirements.txt
```

> **Note:** The folders are currently placeholders and will be populated as the project progresses.

---

# Experiment 1 — LSTM Attitude Estimation (Baseline)

### Objective

Develop and evaluate a baseline LSTM model for estimating Roll (φ), Pitch (θ), and Yaw (ψ) from navigation sensor measurements.

### Description

- Implemented a stacked LSTM model for attitude estimation using navigation sensor data.
- Trained and evaluated the model on six independent datasets containing accelerometer, gyroscope, and magnetometer measurements.
- Applied data preprocessing, Min-Max normalization, sequence generation, and incremental learning to improve temporal prediction.
- Evaluated model performance using **RMSE**, **MAE**, and **R² Score** for Roll, Pitch, and Yaw estimation.
- Analysed the model's ability to generalize across different datasets and identified performance variations under different operating conditions.
- Established a baseline implementation that will be used for comparison with future experiments involving architecture improvements, hyperparameter tuning, and advanced deep learning techniques.

---

**Status:** ✅ Completed (Baseline Experiment)

**Next Experiment:** Improving generalization through architecture modifications, hyperparameter tuning, and multi-dataset training.



experiment with these changes:

1. Change `timestep` values: 2, 5, 10, 20, 50
2. Change LSTM units: 50, 64, 100, 128
3. Add more LSTM layers
4. Try `GRU` instead of `LSTM`
5. Try `Bidirectional LSTM`
6. Change `Dropout` rate: 0.1, 0.2, 0.3, 0.5
7. Change optimizer: `Adam`, `RMSprop`, `SGD`
8. Change learning rate
9. Change batch size: 16, 32, 64, 128
10. Change epochs: 20, 50, 100
11. Add `EarlyStopping`
12. Add `ModelCheckpoint`
13. Add validation split
14. Use train-validation-test split instead of only train-test
15. Normalize both input and output data
16. Try `StandardScaler` instead of `MinMaxScaler`
17. Train separately on each dataset
18. Combine all 6 datasets and train one model
19. Train on 5 datasets and test on 1 dataset
20. Try incremental learning with different chunk sizes: 500, 1000, 3000
21. Change incremental training epochs
22. Freeze some layers during incremental learning
23. Compare offline learning vs incremental learning
24. Add noise to data and test robustness
25. Remove noisy/unimportant features
26. Add derived features like velocity, angular rate, error difference
27. Try predicting sensor error instead of direct attitude values
28. Try multi-step prediction
29. Compare RMSE, MAE, MSE, R² for all experiments
30. Plot residual errors for roll, pitch, yaw
31. Compare actual vs predicted plots for each experiment
32. Tune model separately for yaw because yaw error is higher
33. Try CNN-LSTM model
34. Try Attention-LSTM model
35. Try Transformer-based time-series model
36. Save best model and compare with baseline methods
37. Compare LSTM output with traditional Kalman filter output
38. Use cross-validation across all 6 datasets
39. Check overfitting using training loss vs validation loss
40. Try different random seeds and average the results
