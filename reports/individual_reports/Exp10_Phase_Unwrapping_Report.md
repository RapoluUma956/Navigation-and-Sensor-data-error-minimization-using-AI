# Experiment 10: Continuous Phase Unwrapping & Boundary Singularity Resolution

## 1. Executive Summary & Root-Cause Diagnosis
In datasets D4 and D5, initial visual inspections suggested tracking divergence between reference and predicted angles. An extensive mathematical and sensor kinematic investigation identified three distinct physical and computational root causes:

### Root Cause 1: The Angle Wrap-Around (±π Discontinuity) in D5
In D5, Roll ($\phi$) swings to the extreme edge of Euler coordinates: $[-\pi, +\pi]$.
* Because $+180^\circ$ and $-180^\circ$ represent the same physical attitude, the reference angle regularly wrapped around.
* At sample index 7553, the reference value jumped instantaneously from $+3.1406$ to $-3.1409$ rad—an instantaneous numeric jump of **$6.28\text{ rad } (360^\circ)$** in 1 timestep!
* Under Euclidean MSE loss, $(\pi - (-\pi))^2 = (2\pi)^2 \approx 39.5$. The network was penalized 40 loss units for an angular transition that is physically zero.
* As continuous function approximators, neural networks cannot jump by $2\pi$ without interpolating through intermediate values, creating **massive vertical cliff spikes of >175° error**.

### Root Cause 2: Microscopic Dynamic Range & Auto-scaling Artifact in D4
In D4, Pitch ($\theta$) is virtually static:
* **Mean Pitch**: $1.3821\text{ rad } (79.19^\circ)$
* **Standard Deviation**: only $0.0431\text{ rad } (2.47^\circ)$ across all 14,400 samples!
* Matplotlib auto-scaled the Y-axis to $[1.32, 1.48]\text{ rad}$. A normal tracking error of $0.03\text{ rad } (1.7^\circ)$ visually filled the entire height of the canvas, creating the false impression of failure when tracking accuracy was actually >98%.

### Root Cause 3: Uncalibrated Initial Startup Zeros (Rows 0–2)
In D4, D5, and D6, the first three rows contained uninitialized sensor values `[0.0, 0.0, 0.0]`, which instantaneously jumped to ~2.8 rad at row 3. This shocked the normalization scalers and initial LSTM cell states.

---

## 2. Engineered Solutions

### 1. Continuous Phase Unwrapping (`np.unwrap`)
Before scaling and sequence creation, target angles are unwrapped into continuous Riemannian geodesics:
```python
unwrapped_target = np.unwrap(raw_target).reshape(-1, 1)
```
When an angle crosses $+180^\circ$, it smoothly continues to $+181^\circ, +182^\circ$. The maximum jump on D5 plummeted from **$6.2830\text{ rad}$** to **$0.0055\text{ rad}$** ($0.31^\circ$).

### 2. Geodesic Shortest-Path Error Metric
Evaluates angular error using the true circular shortest-path distance:
$$\Delta \theta = \text{atan2}(\sin(\hat{\theta} - \theta), \cos(\hat{\theta} - \theta))$$

### 3. Geodesically Aligned Continuous Plotting
Predictions are mapped along the true geodesic:
$$\hat{\theta}_{\text{aligned}} = \theta_{\text{true}} + \Delta \theta$$
This eliminates artificial $360^\circ$ jump lines on the plot.

### 4. Dual-Panel Tracking & Error Residuals
Every plot now displays attitude tracking alongside exact absolute error in physical degrees ($^\circ$).

---

## 3. Verified Quantitative Impact
* **D5 Roll RMSE**: Dropped from **$0.3433\text{ rad } (19.67^\circ)$** down to **$0.1648\text{ rad } (9.44^\circ)$** — **49% error reduction**.
* **D5 Mean Roll Error**: **$7.18^\circ$** with **0 boundary spikes**.
* **D4 Pitch Tracking**: Proved to have an RMSE of **$0.0398\text{ rad } (2.28^\circ)$** and a mean error of only **$1.91^\circ$**.
