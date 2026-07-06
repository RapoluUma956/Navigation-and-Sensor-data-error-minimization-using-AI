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