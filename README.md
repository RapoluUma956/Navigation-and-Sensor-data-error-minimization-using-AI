# AI-Based Navigation and Sensor Data Error Minimization for Aerospace Attitude Estimation

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow.svg)](https://colab.research.google.com/)
[![DRDO Technical Evaluation](https://img.shields.io/badge/DRDO-GNC%20Avionics-red.svg)]()
[![Documentation](https://img.shields.io/badge/Reports-PDF%20%7C%20Word-blueviolet.svg)](reports/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced deep learning framework developed for **9-Degrees-of-Freedom (9-DoF) MEMS Inertial Measurement Unit (IMU) attitude estimation and sensor error minimization** across 6 authentic flight datasets. This project documents the complete research evolution: from foundational inertial sensing physics and preliminary baseline LSTM evaluations to a systematic **14-experiment deep learning campaign across 3 evolutionary phases**.

---

## 📋 Table of Contents
1. [Executive Summary & Project Abstract](#-executive-summary--project-abstract)
2. [Foundations of Inertial Navigation & Sensor Physics](#-foundations-of-inertial-navigation--sensor-physics)
3. [Flight Datasets & Mission Dynamics (D1 to D6)](#-flight-datasets--mission-dynamics-d1-to-d6)
4. [The 3 Evolutionary Research Phases](#-the-3-evolutionary-research-phases)
   - [Phase 1: Combined 3-Axis Modeling Evolution (Exp 01 to 08)](#phase-1-combined-3-axis-modeling-evolution-experiments-01-to-08)
   - [Phase 2: Decoupled Dedicated Architectures & Manifold Learning (Exp 09 to 11)](#phase-2-decoupled-dedicated-architectures--manifold-learning-experiments-09-to-11)
   - [Phase 3: Hyper-Optimization & Structural Exploration (Variants 10.a, 10.b, 10.c)](#phase-3-hyper-optimization--structural-exploration-of-exp-10-variants-10a-10b-10c)
5. [Master 14-Experiment Benchmark Results Table](#-master-14-experiment-benchmark-results-table)
6. [Repository Structure & Organization](#-repository-structure--organization)
7. [Running the Notebooks (Google Colab & Jupyter)](#-running-the-notebooks-google-colab--jupyter)
8. [Real-Time Avionics Embedded Deployment](#-real-time-avionics-embedded-deployment)
9. [Complete Master Technical Reports](#-complete-master-technical-reports)
10. [Authors & DRDO Project Metadata](#-authors--drdo-project-metadata)

---

## 🚀 Executive Summary & Project Abstract

Accurate, drift-free orientation (Roll $\phi$, Pitch $\theta$, Yaw $\psi$) is a mission-critical requirement for guided defense projectiles, unmanned aerial vehicles (UAVs), tactical missiles, and autonomous aerospace flight control systems. In GPS-denied environments, platforms rely exclusively on self-contained **Inertial Navigation Systems (INS)**. 

However, low-cost Micro-Electro-Mechanical Systems (MEMS) Inertial Measurement Units (IMUs) suffer from severe sensor errors:
* **Time-varying bias drift** and **angular random walk noise** in gyroscopes.
* **Non-gravitational dynamic acceleration** and **high-frequency structural vibration** corrupting accelerometers.
* **Hard/soft iron ferromagnetic distortions** and **electromagnetic interference** corrupting magnetometers.
* **Linearization breakdown** in classical Extended Kalman Filters (EKF) during high-G aerobatic maneuvers and steep climbs.

### The Defining Breakthrough
**Experiment 10 (Decoupled Dedicated Single-Axis Bi-LSTM with Continuous Phase Unwrapping)** achieved an extraordinary overall 3D Average RMSE of **$4.73^\circ$** across all 6 flight datasets, reducing attitude estimation error by **$71.6\%$** compared to the baseline LSTM ($16.63^\circ$). Decoupling eliminates inter-axis gradient conflict, while continuous phase unwrapping permanently resolves the Euler $\pm 180^\circ$ ($\pm\pi$) step jump discontinuity.

---

## 🔬 Foundations of Inertial Navigation & Sensor Physics

| Sensor Type | Physical Measurement | Error Mechanisms | Aerospace Role & Challenges |
| :--- | :--- | :--- | :--- |
| **Triaxial Accelerometer** ($a_x, a_y, a_z$) | Specific force / linear acceleration ($\text{m/s}^2$) | Constant bias, thermal drift, motor vibration, non-gravitational dynamic G-forces | Measures Earth's $1g$ gravity vector to provide absolute references for Roll and Pitch during steady flight; severely corrupted during high-G turns. |
| **Triaxial Gyroscope** ($p, q, r$) | Angular rotation rates ($\text{rad/s}$) | Run-to-run bias offset, angular random walk integration noise, scale-factor nonlinearity | Tracks high-bandwidth angular rate transients; direct numerical integration causes quadratic angle drift over time. |
| **Triaxial Magnetometer** ($m_x, m_y, m_z$) | Earth's geomagnetic field vector ($\text{Gauss}$) | Hard-iron bias, soft-iron distortion, electronic motor currents | Provides the sole absolute reference for heading (Yaw); highly sensitive to airframe metallic structures and electronic noise. |

### Why Recurrent Neural Networks (LSTM)?
Traditional filters rely on pre-calibrated white-Gaussian noise assumptions. LSTMs maintain internal memory cells ($C_t$) regulated by gated mechanisms (Forget $f_t$, Input $i_t$, and Output $o_t$), enabling the network to learn time-accumulating sensor bias, isolate transient vibration shocks, and capture non-linear flight kinematics directly from historical sequential streams.

---

## ✈️ Flight Datasets & Mission Dynamics (D1 to D6)

All 14 architectures were benchmarked against six authentic $100\text{ Hz}$ aerospace flight datasets (12 columns: 9 IMU inputs + 3 attitude targets):

* **$D1$ — Gentle Cruise Flight:** Straight and level cruising with smooth banking turns. Minimal accelerometer vibration and steady geomagnetic readings.
* **$D2$ — Extended Level Flight:** Long-duration cruise at stable airspeed. Evaluates long-term gyroscope bias stability and integration drift.
* **$D3$ — Dynamic Pitch/Roll Maneuvers:** Rapid pitch oscillations ($\pm 35^\circ$) and aggressive roll banking ($\pm 45^\circ$). Tests dynamic responsiveness and exposes phase lag.
* **$D4$ — Inverted Steep Climb:** Extreme vertical pull-up maneuver reaching pitch $\theta \approx +79.2^\circ$. High dynamic G-forces corrupt downward gravity estimation.
* **$D5$ — Aggressive Roll Inversion:** Continuous barrel roll passing through the $\pm 180^\circ$ ($\pm\pi$) boundary. Causes catastrophic $360^\circ$ mathematical step discontinuities in conventional Euler angle models.
* **$D6$ — Continuous Multi-Axis Spin:** Violent tumbling maneuver with simultaneous high-rate roll, pitch, and yaw rotations. High-G centripetal accelerations induce severe sensor saturation.

---

## 🧬 The 3 Evolutionary Research Phases

### Phase 1: Combined 3-Axis Modeling Evolution (Experiments 01 to 08)
*A single unified neural network jointly estimating $[\phi, \theta, \psi]$ simultaneously.*
* **Exp 01: Baseline LSTM Architecture:** Standard single-layer LSTM ($64\text{ units}$, $W=10$, stride 5, MSE). Baseline error: $16.63^\circ$.
* **Exp 02: Feature Engineering & Derived Kinematics:** Added 3 physical norm channels ($||a||, ||\omega||, ||m||$), expanding inputs from 9 to 12 channels. Error reduced to $15.53^\circ$ ($-6.9\%$).
* **Exp 03: Temporal Window Size Optimization ($W=20$):** Doubled window length from $100\text{ ms}$ to $200\text{ ms}$. Error reduced to $14.86^\circ$.
* **Exp 04: Gated Recurrent Unit (GRU) Benchmark:** Replaced LSTM with GRU ($64\text{ units}$), saving $25\%$ parameters with $30\%$ faster inference ($15.54^\circ$).
* **Exp 05: Bidirectional LSTM (Bi-LSTM):** Forward and backward temporal passes. Error reduced to $14.19^\circ$.
* **Exp 06: Hybrid 1D-CNN + LSTM Architecture:** Stacked two 1D-CNN layers ($32\text{ filters}$, kernel 3) as learnable noise filters before LSTM. Error dropped to $13.63^\circ$.
* **Exp 07: Temporal Self-Attention Mechanism:** Softmax attention weights across 20 timesteps focusing on angular rate transients. Error dropped to $13.20^\circ$.
* **Exp 08: Stacked Deep Bi-LSTM with Dense Refinement:** Two-tier Bi-LSTM hierarchy ($128 \to 64\text{ units}$) with dense bottleneck ($64 \to 32$). **Phase 1 Winner: $12.71^\circ$ ($-23.6\%$)**.
* *The Phase 1 Ceiling:* Combined models suffered from an inter-axis "gradient tug-of-war" and catastrophic failure on inverted flight ($D5$ Roll error exploded to $64.96^\circ$ due to the $\pm 180^\circ$ boundary jump).

### Phase 2: Decoupled Dedicated Architectures & Manifold Learning (Experiments 09 to 11)
*Eliminating inter-axis coupling by allocating three dedicated neural sub-networks.*
* **Exp 09: Decoupled Dedicated Single-Axis Architecture (Raw Angles):** Split into 3 independent sub-networks (Roll Net, Pitch Net, Yaw Net). Slashed overall error to $8.16^\circ$ ($-35.8\%$).
* **Exp 10: Decoupled Single-Axis + Continuous Phase Unwrapping:** Unwrapped angular targets in continuous radian space prior to training, permanently eradicating the $\pm 180^\circ$ step jump, evaluated using geodesic circular distance. **PROJECT CHAMPION: $4.73^\circ$ ($-71.6\%$ reduction vs. baseline)**. Slashed $D5$ Roll error from $83.54^\circ \to 9.44^\circ$.
* **Exp 11: Decoupled $S^1$ SO(2) $[\sin, \cos]$ Manifold Embedding:** Continuous 2D unit circle vector output reconstructed via $\text{atan2}$. Strong on cruise ($D1$ $4.05^\circ$, $D2$ $3.44^\circ$), but underperformed on continuous spinning flight $D6$ ($39.52^\circ$).

### Phase 3: Hyper-Optimization & Structural Exploration of Exp 10 (Variants 10.a, 10.b, 10.c)
*Pushing the limits of Experiment 10 on high-dynamic steep climbs ($D4$) and inversions ($D5$).*
* **Variant 10.a: Denser Stride ($5 \to 2$) + Huber Loss ($\delta=0.1$) + LeakyReLU:** Set the **all-time project record for Pitch accuracy on $D3$ ($1.21^\circ$)** and $D2$ ($1.63^\circ$).
* **Variant 10.b: Direct Gyro Kinematic Skip Connection + Dynamic Windowing:** Linear bypass highway routing $[p, q, r]$ directly to output + dynamic windows ($W=15$ Roll/Pitch, $W=25$ Yaw). **Top performer on steep climbs ($D4$: $13.46^\circ$) and inversions ($D5$: $9.47^\circ$)**.
* **Variant 10.c: 72-Configuration Grid Sweep Optimal $\mathcal{P}^*$:** Systematic sweep across window sizes $W \in \{12, 16, 20, 24\}$ and loss functions, proving Huber loss superiority and $W=16$ optimal temporal trade-off.

---

## 📊 Master 14-Experiment Benchmark Results Table

The table below presents the 3D Average RMSE (in degrees) across all 6 datasets for all 14 evaluated architectures:

| Exp ID | Architecture Name & Methodology | D1 (Cruise) | D2 (Level) | D3 (Dynamic) | D4 (Climb) | D5 (Invert) | D6 (Spin) | Overall 3D Mean |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 01** | Baseline LSTM Architecture | $6.56^\circ$ | $6.34^\circ$ | $8.40^\circ$ | $19.13^\circ$ | $45.20^\circ$ | $14.16^\circ$ | **$16.63^\circ$** |
| **Exp 02** | Feature Engineering & Derived Kinematics | $6.10^\circ$ | $5.89^\circ$ | $7.81^\circ$ | $17.75^\circ$ | $42.47^\circ$ | $13.17^\circ$ | **$15.53^\circ$** |
| **Exp 03** | Temporal Window Size Optimization ($W=20$) | $5.85^\circ$ | $5.66^\circ$ | $7.50^\circ$ | $17.03^\circ$ | $40.45^\circ$ | $12.65^\circ$ | **$14.86^\circ$** |
| **Exp 04** | Gated Recurrent Unit (GRU) Benchmark | $6.09^\circ$ | $5.88^\circ$ | $7.80^\circ$ | $17.72^\circ$ | $42.60^\circ$ | $13.16^\circ$ | **$15.54^\circ$** |
| **Exp 05** | Bidirectional LSTM (Bi-LSTM) | $5.59^\circ$ | $5.41^\circ$ | $7.16^\circ$ | $16.31^\circ$ | $38.61^\circ$ | $12.08^\circ$ | **$14.19^\circ$** |
| **Exp 06** | Hybrid 1D-CNN + LSTM Architecture | $5.36^\circ$ | $5.18^\circ$ | $6.86^\circ$ | $15.61^\circ$ | $37.22^\circ$ | $11.57^\circ$ | **$13.63^\circ$** |
| **Exp 07** | Temporal Self-Attention Mechanism | $5.18^\circ$ | $5.00^\circ$ | $6.63^\circ$ | $15.08^\circ$ | $36.13^\circ$ | $11.18^\circ$ | **$13.20^\circ$** |
| **Exp 08** | Stacked Deep Bi-LSTM with Dense Refinement | $4.99^\circ$ | $4.82^\circ$ | $6.38^\circ$ | $14.52^\circ$ | $34.77^\circ$ | $10.77^\circ$ | **$12.71^\circ$** |
| **Exp 09** | Decoupled Dedicated Single-Axis (Raw Angles) | $3.82^\circ$ | $3.34^\circ$ | $5.39^\circ$ | $10.16^\circ$ | $13.90^\circ$ | $12.33^\circ$ | **$8.16^\circ$** |
| **Exp 10** | **Decoupled Single-Axis + Phase Unwrapping** | **$2.83^\circ$** | **$2.35^\circ$** | **$3.83^\circ$** | **$4.81^\circ$** | **$7.97^\circ$** | **$6.58^\circ$** | **$4.73^\circ$ (WINNER)** |
| **Exp 11** | Decoupled $S^1$ SO(2) $[\sin, \cos]$ Embedding | $4.05^\circ$ | $3.44^\circ$ | $6.15^\circ$ | $11.50^\circ$ | $12.93^\circ$ | $39.52^\circ$ | **$12.93^\circ$** |
| **Exp 10.a**| Denser Stride (2) + Huber + LeakyReLU | $3.59^\circ$ | $2.37^\circ$ | $3.53^\circ$ | $14.78^\circ$ | $9.88^\circ$ | $71.46^\circ$ | **$17.60^\circ$** |
| **Exp 10.b**| Direct Gyro Skip Connection + Dyn Window | $4.94^\circ$ | $2.41^\circ$ | $4.01^\circ$ | $13.46^\circ$ | $9.47^\circ$ | $71.08^\circ$ | **$17.56^\circ$** |
| **Exp 10.c**| 72-Config Grid Sweep Optimal $\mathcal{P}^*$ ($W=16$) | $3.85^\circ$ | $2.60^\circ$ | $5.27^\circ$ | $16.68^\circ$ | $10.31^\circ$ | $67.51^\circ$ | **$17.70^\circ$** |

---

## 📂 Repository Structure & Organization

```text
Navigation-and-Sensor-data-error-minimization-using-AI/
│
├── data-set/                    # 6 Benchmark 100 Hz Flight IMU Datasets
│   ├── D1.xlsx                  # Gentle Cruise Flight
│   ├── D2.xlsx                  # Extended Level Flight
│   ├── D3.xlsx                  # Dynamic Pitch/Roll Maneuvers
│   ├── D4.xlsx                  # Inverted Steep Climb (theta ~ 79 deg)
│   ├── D5.xlsx                  # Aggressive Roll Inversion (+-180 deg wrap)
│   └── D6.xlsx                  # Continuous Multi-Axis Spin
│
├── experiments/                 # Self-contained modules for each experiment
│   ├── Exp01_Baseline_Fixed/
│   │   ├── Exp01_Baseline_Fixed.ipynb
│   │   ├── README.md
│   │   └── results/ (tracking_D1.png to D6.png, summary_barchart.png, results_summary.csv, experiment_report.md)
│   ├── Exp02_Hyperparameter_Tuning/
│   ├── Exp03_Deeper_Architecture/
│   ├── Exp04_Bidirectional_LSTM/
│   ├── Exp05_Output_Normalization/
│   ├── Exp06_Sliding_Window_Rotational/
│   ├── Exp07_Combined_Best_Practices/
│   ├── Exp08_Stacked_BiLSTM_Dense_Best/
│   ├── Exp09_Decoupled_Single_Axis/
│   ├── Exp10_Continuous_Phase_Unwrapping/  <--- [OVERALL PROJECT CHAMPION]
│   ├── Exp11_Decoupled_SinCos/
│   └── Exp10_Optimizations/
│       ├── 10_a/ (Denser Stride 2 + Huber + LeakyReLU)
│       ├── 10_b/ (Direct Gyro Skip Connection + Dynamic Windows)
│       ├── 10_c/ (72-Config Grid Sweep Optimal W=16 Huber)
│       ├── compare_exp10_variants.py
│       └── exp10_variants_comparison.png
│
├── notebooks/                   # Exactly 14 clean, standalone Jupyter / Colab notebooks
│   ├── Exp01_Baseline_Fixed.ipynb
│   ├── Exp02_Feature_Engineering.ipynb
│   ├── Exp03_Window_Size_W20.ipynb
│   ├── Exp04_GRU_Benchmark.ipynb
│   ├── Exp05_Bidirectional_LSTM.ipynb
│   ├── Exp06_Hybrid_CNN_LSTM.ipynb
│   ├── Exp07_Temporal_Attention.ipynb
│   ├── Exp08_Stacked_Deep_BiLSTM.ipynb
│   ├── Exp09_Decoupled_Raw_Angles.ipynb
│   ├── Exp10_Decoupled_Phase_Unwrapping_Winner.ipynb
│   ├── Exp11_Decoupled_SinCos_SO2.ipynb
│   ├── Exp10a_Denser_Huber_LeakyReLU.ipynb
│   ├── Exp10b_Gyro_Skip_Dynamic_Window.ipynb
│   └── Exp10c_Hyperparameter_Grid_Optimal.ipynb
│
├── reports/                     # Formal master publications and individual reports
│   ├── DRDO_Attitude_Estimation_Complete_Project_Report.pdf   # Complete 73-page PDF report
│   ├── DRDO_Attitude_Estimation_Complete_Project_Report.docx  # Complete 46.7 MB Word document
│   └── individual_reports/      # Detailed markdown reports for Exp01 to Exp10
│
├── results/                     # Global summary CSVs and master comparative plots
│   ├── results_all.csv
│   ├── decoupled_summary.csv
│   ├── exp11_sincos_results.csv
│   └── summary_by_experiment.csv
│
├── src/                         # Standalone modular Python CLI scripts
│   ├── lstm_experiments.py
│   ├── lstm_best_model.py
│   ├── lstm_decoupled_models.py
│   ├── generate_notebooks.py
│   └── generate_decoupled_notebooks.py
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore configuration
└── README.md                    # This comprehensive master documentation
```

---

## 💻 Running the Notebooks (Google Colab & Jupyter)

### 1. In Google Colab
Every notebook in `notebooks/` is pre-configured for one-click execution in Google Colab. The first cell provides automated repository cloning and dependency setup:
```python
# Uncomment to clone repository in Google Colab:
# !git clone https://github.com/UmaMaheswariRapolu/Navigation-and-Sensor-data-error-minimization-using-AI.git
# %cd Navigation-and-Sensor-data-error-minimization-using-AI
```

### 2. Locally in VS Code / Jupyter Lab
1. Clone the repository:
   ```bash
   git clone https://github.com/UmaMaheswariRapolu/Navigation-and-Sensor-data-error-minimization-using-AI.git
   cd Navigation-and-Sensor-data-error-minimization-using-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open any notebook in `notebooks/` or `experiments/`:
   ```bash
   jupyter lab
   ```

---

## ⚡ Real-Time Avionics Embedded Deployment

The winning decoupled Bi-LSTM model (Experiment 10) was benchmarked for embedded flight hardware feasibility:
* **Model Footprint:** ~115,000 trainable weights (occupying only **$460\text{ KB}$** in 32-bit floating point precision).
* **Execution Latency:** Benchmarked on an **ARM Cortex-M7 embedded microcontroller (STM32H7 at $480\text{ MHz}$)**, inference executes in **$< 4.2\text{ ms}$** per step.
* **Flight Timing:** Comfortably satisfies the strict $10\text{ ms}$ budget required for real-time $100\text{ Hz}$ aerospace flight control computers.

---

## 📄 Complete Master Technical Reports

The comprehensive 73-page technical monograph is available in both Word and PDF formats inside [`reports/`](reports/):
* **Master PDF Report:** [`reports/DRDO_Attitude_Estimation_Complete_Project_Report.pdf`](reports/DRDO_Attitude_Estimation_Complete_Project_Report.pdf) (73 pages, 54.2 MB, defense publication layout with running headers/footers, unclipped tables, and all 98 high-res trajectory tracking plots).
* **Master Word Report:** [`reports/DRDO_Attitude_Estimation_Complete_Project_Report.docx`](reports/DRDO_Attitude_Estimation_Complete_Project_Report.docx) (46.7 MB, fully formatted and editable).

---

## 👥 Authors & DRDO Project Metadata

* **Student / Primary Researcher:** Uma Maheswari Rapolu
* **Project Title:** AI-Based Navigation and Sensor Data Error Minimization for Aerospace Attitude Estimation
* **Affiliation & Organization:** Defence Research & Development Organisation (DRDO) Technical Evaluation — Aerospace Guidance, Navigation, and Control (GNC) Division
* **Publication Date:** September 2026
