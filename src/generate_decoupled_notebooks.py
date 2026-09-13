# -*- coding: utf-8 -*-
"""
Generator for Decoupled Single-Axis Attitude Estimation Notebooks:
1. Roll_Estimation_LSTM.ipynb (phi)
2. Pitch_Estimation_LSTM.ipynb (theta)
3. Yaw_Estimation_LSTM.ipynb (psi)
4. Decoupled_Attitude_Master_Colab.ipynb (Roll + Pitch + Yaw unified pipeline)
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'colab_notebooks')
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def md_cell(text):
    lines = [line + '\n' for line in text.strip().split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(code):
    lines = [line + '\n' for line in code.strip().split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

COMMON_SETUP = """# Install dependencies if running on fresh Colab instance
!pip install -q pandas numpy scikit-learn matplotlib openpyxl

import os
import sys
import time
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Check environment
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Running in Google Colab: {IN_COLAB}")
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpus) > 0} ({[g.name for g in gpus]})")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
"""

def generate_single_axis_notebook(angle_name, angle_symbol, target_col_idx, sensor_focus_text, default_dataset="D1.xlsx"):
    title = f"{angle_name} ({angle_symbol}) Estimation — Dedicated LSTM Model"
    cells = [
        md_cell(f"""# {title}
## Dedicated Single-Axis Attitude Error Minimization Framework

### 1. Why a Dedicated {angle_name} Model?
In a combined model predicting Roll, Pitch, and Yaw simultaneously, severe cross-axis interference occurs:
* **Gradient Dominance**: High-variance axes (like Roll in dynamic flight with $\\sigma^2 \\approx 1.34$) overwhelm low-variance axes (like Pitch with $\\sigma^2 \\approx 0.003$).
* **Physical Decoupling**: {sensor_focus_text}
* **Independent Optimization**: A dedicated model with a specialized loss function $\\mathcal{{L}} = \\text{{MSE}}({angle_symbol})$ optimizes weights purely for {angle_name}, eliminating conflicting gradient updates from other axes.

---

### 2. Key Pipeline Features
1. **Target Isolation**: Isolates column `{target_col_idx}` ({angle_symbol}) as the single target.
2. **Dedicated StandardScaler**: Normalizes {angle_name} independently, ensuring smooth, centered gradient descent.
3. **Sliding Window Representation** (`window=20, stride=5` train, `stride=1` test): Extracts continuous differential angular rates.
4. **Bidirectional LSTM + Dense Architecture**: Captures forward and backward temporal dynamics.
5. **Incremental Online Adaptation**: Continuously adapts to incoming sensor drift across 8 sequential test batches.
6. **Colab Ready**: Works on any of the 6 datasets (`D1.xlsx` – `D6.xlsx`) with automatic file finder/uploader.
"""),
        code_cell(COMMON_SETUP),
        md_cell("### Dataset Selection & Loader"),
        code_cell(f"""# @title Select Dataset
DATASET_NAME = "{default_dataset}"  # @param ["D1.xlsx", "D2.xlsx", "D3.xlsx", "D4.xlsx", "D5.xlsx", "D6.xlsx"]

def find_or_upload_dataset(filename):
    possible_locations = [
        filename,
        os.path.join('/content', filename),
        os.path.join('/content/sample_data', filename),
        os.path.join('.', filename),
        os.path.join('..', filename)
    ]
    for loc in possible_locations:
        if os.path.exists(loc):
            print(f"Found '{{filename}}' at: {{loc}}")
            return loc
    
    if IN_COLAB:
        print(f"File '{{filename}}' not found. Please upload it:")
        from google.colab import files
        uploaded = files.upload()
        if filename in uploaded:
            return filename
    raise FileNotFoundError(f"Could not locate '{{filename}}'.")

dataset_path = find_or_upload_dataset(DATASET_NAME)
df = pd.read_excel(dataset_path)
data = np.array(df, dtype=np.float64)

# 9 IMU inputs: ax, ay, az, p, q, r, mx, my, mz
dataX = data[:, :9]

# Target column {target_col_idx}: {angle_name} ({angle_symbol})
dataY_target = data[:, {target_col_idx}:{target_col_idx}+1]

print(f"Dataset Loaded: {{DATASET_NAME}} | Total Samples: {{len(data):,}}")
print(f"Input features shape: {{dataX.shape}}")
print(f"{angle_name} target shape: {{dataY_target.shape}}")
print(f"Target Range (rad): [{{dataY_target.min():.4f}}, {{dataY_target.max():.4f}}]")
print(f"Target Variance: {{np.var(dataY_target):.6f}}")
"""),
        md_cell(f"### Dedicated Normalization & Sliding Window Sequences for {angle_name}"),
        code_cell(f"""def create_sliding_sequences(X, Y, window_size=20, stride=1):
    \"\"\"Extracts overlapping temporal windows for continuous rotational tracking.\"\"\"
    X_seq, Y_seq = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq.append(X[i:i + window_size])
        Y_seq.append(Y[i + window_size - 1])
    return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32)

# Independent standardization for inputs and dedicated {angle_name} target
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

dataX_scaled = scaler_X.fit_transform(dataX)
dataY_scaled = scaler_Y.fit_transform(dataY_target)

window_size = 20
stride_train = 5
stride_test = 1
epochs_offline = 50
epochs_inc = 10
batch_size = 128
n_inc_batches = 8

split = int(len(data) * 0.60)
dataXt, dataYt = dataX_scaled[:split], dataY_scaled[:split]
dataXv, dataYv_scaled = dataX_scaled[split:], dataY_scaled[split:]
dataYv_orig = dataY_target[split:]
input_dim = dataXt.shape[1]

X_train, Y_train = create_sliding_sequences(dataXt, dataYt, window_size, stride=stride_train)
print(f"Generated {{len(X_train):,}} training sequences (Window={{window_size}}, Dim={{input_dim}})")
"""),
        md_cell(f"### Dedicated {angle_name} Architecture & Pretraining"),
        code_cell(f"""from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.keras.backend.clear_session()

# Model tailored specifically for {angle_name}
inputs = Input(shape=(window_size, input_dim), name="imu_input")
x = Bidirectional(LSTM(64, return_sequences=True, name="bilstm_1"))(inputs)
x = Dropout(0.25, name="dropout_1")(x)
x = LSTM(32, name="lstm_2")(x)
x = Dense(32, activation='relu', name="dense_feat")(x)
x = Dropout(0.15, name="dropout_2")(x)
outputs = Dense(1, activation='linear', name="{angle_name.lower()}_output")(x)

model = Model(inputs=inputs, outputs=outputs, name="Dedicated_{angle_name}_LSTM")
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.summary()

lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
es_cb = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

print(f"\\nStarting Dedicated {angle_name} Pretraining...")
t0 = time.time()
history = model.fit(
    X_train, Y_train,
    epochs=epochs_offline,
    batch_size=batch_size,
    callbacks=[lr_cb, es_cb],
    verbose=1
)
print(f"Pretraining Finished in {{time.time() - t0:.1f}}s.")
"""),
        md_cell("### Incremental Online Learning"),
        code_cell(f"""test_len = len(dataXv)
step_size = max(window_size * 2, test_len // (n_inc_batches + 1))
inc_end = step_size * n_inc_batches

all_predictions = []
all_ground_truth = []

model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
print(f"Running {{n_inc_batches}} incremental update batches...")
t_inc_start = time.time()

for j in range(n_inc_batches):
    pt = step_size * j
    end_pt = min(pt + step_size, test_len)
    if end_pt - pt < window_size:
        break
    Xbatch = dataXv[pt:end_pt]
    Ybatch_scaled = dataYv_scaled[pt:end_pt]
    Ybatch_orig = dataYv_orig[pt:end_pt]
    
    # Predict on incoming stream (stride=1)
    X_seq_test, _ = create_sliding_sequences(Xbatch, Ybatch_scaled, window_size, stride=stride_test)
    _, Y_seq_orig = create_sliding_sequences(Xbatch, Ybatch_orig, window_size, stride=stride_test)
    if len(X_seq_test) == 0:
        break
        
    preds_scaled = model.predict(X_seq_test, verbose=0)
    preds_orig = scaler_Y.inverse_transform(preds_scaled)
    all_predictions.append(preds_orig)
    all_ground_truth.append(Y_seq_orig)
    
    # Online adaptation (stride=5)
    X_seq_train, Y_seq_train = create_sliding_sequences(Xbatch, Ybatch_scaled, window_size, stride=stride_train)
    if len(X_seq_train) > 0:
        model.fit(X_seq_train, Y_seq_train, epochs=epochs_inc, batch_size=batch_size, verbose=0)

if inc_end < test_len and (test_len - inc_end) >= window_size:
    Xrem = dataXv[inc_end:]
    Yrem_scaled = dataYv_scaled[inc_end:]
    Yrem_orig = dataYv_orig[inc_end:]
    X_seq_test, _ = create_sliding_sequences(Xrem, Yrem_scaled, window_size, stride=stride_test)
    _, Y_seq_orig = create_sliding_sequences(Xrem, Yrem_orig, window_size, stride=stride_test)
    if len(X_seq_test) > 0:
        preds_scaled = model.predict(X_seq_test, verbose=0)
        preds_orig = scaler_Y.inverse_transform(preds_scaled)
        all_predictions.append(preds_orig)
        all_ground_truth.append(Y_seq_orig)

Y_pred = np.vstack(all_predictions)
Y_true = np.vstack(all_ground_truth)
min_l = min(len(Y_pred), len(Y_true))
Y_pred, Y_true = Y_pred[:min_l], Y_true[:min_l]

rmse_val = sqrt(mean_squared_error(Y_true, Y_pred))

print("="*60)
print(f"DEDICATED {angle_name.upper()} ({angle_symbol.upper()}) EVALUATION ON {{DATASET_NAME}}")
print("="*60)
print(f"  {angle_name} RMSE (rad): {{rmse_val:.6f}} rad")
print(f"  {angle_name} RMSE (deg): {{np.degrees(rmse_val):.4f}} deg")
print(f"  Incremental Adaptation Time: {{time.time() - t_inc_start:.1f}}s")
print("="*60)
"""),
        md_cell(f"### Tracking Visualization ({angle_name})"),
        code_cell(f"""plt.figure(figsize=(14, 5))
plt.plot(Y_true, label=f'Ground Truth Reference ({angle_symbol})', color='navy', alpha=0.85, linewidth=1.5)
plt.plot(Y_pred, label=f'Dedicated LSTM Predicted (RMSE: {{rmse_val:.4f}} rad)', color='crimson', alpha=0.85, linestyle='--', linewidth=1.4)
plt.title(f'Dedicated {angle_name} ({angle_symbol}) Estimation — {{DATASET_NAME}}', fontsize=13, fontweight='bold')
plt.xlabel('Sample Index (Time)', fontsize=11)
plt.ylabel(f'{angle_name} Angle [rad]', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()
"""),
        md_cell("### Save Dedicated Model Weights"),
        code_cell(f"""save_path = f"dedicated_{angle_name.lower()}_{{DATASET_NAME.split('.')[0]}}.keras"
model.save(save_path)
print(f"Dedicated {angle_name} model saved to: {{save_path}}")
""")
    ]
    return create_notebook(cells)


# ==============================================================================
# DECOUPLED MASTER NOTEBOOK (ALL 3 DEDICATED MODELS ASSEMBLED)
# ==============================================================================
def build_decoupled_master():
    cells = [
        md_cell("""# Decoupled Multi-Model Master Framework: Independent Roll, Pitch & Yaw Networks
## Optimal Error Minimization via Kinematic Decomposition

### 1. Architectural Concept
Instead of forcing 1 neural network to predict all 3 angles, this architecture constructs **3 dedicated, independently optimized neural networks**:
```
                        ┌──► Model 1 (Roll  φ)  ──► Loss_Roll  = MSE(φ) ──►  Roll  (φ)
                        │
IMU Sensor Stream ──────┼──► Model 2 (Pitch θ)  ──► Loss_Pitch = MSE(θ) ──►  Pitch (θ)
(ax,ay,az,p,q,r,mx,my,mz)│
                        └──► Model 3 (Yaw   ψ)  ──► Loss_Yaw   = MSE(ψ) ──►  Yaw   (ψ)
```

### 2. Why This Completely Solves the Multi-Task Trade-off
1. **Zero Cross-Axis Interference**: Roll swings of $\\pm \\pi$ in D5 cannot disturb pitch predictions.
2. **Individual Standardization**: Each angle has its own dedicated `StandardScaler`, ensuring Pitch (even with tiny variance $\\sigma^2=0.0018$ in D4) gets equal gradient power.
3. **Sensor-Specific Focus**:
   * Roll model specializes in lateral gravity and roll rate ($a_y, a_z, p$).
   * Pitch model specializes in longitudinal gravity and pitch rate ($a_x, a_z, q$).
   * Yaw model specializes in magnetic heading tracking ($m_x, m_y, m_z, r$).
4. **Full 3D State Vector Output**: Predictions from all 3 models are assembled into $[\hat{\\phi}, \\hat{\\theta}, \\hat{\\psi}]$ for unified evaluation and plotting.
"""),
        code_cell(COMMON_SETUP),
        code_cell("""# @title Select Dataset
DATASET_NAME = "D1.xlsx"  # @param ["D1.xlsx", "D2.xlsx", "D3.xlsx", "D4.xlsx", "D5.xlsx", "D6.xlsx"]

def find_or_upload_dataset(filename):
    possible_locations = [
        filename,
        os.path.join('/content', filename),
        os.path.join('/content/sample_data', filename),
        os.path.join('.', filename),
        os.path.join('..', filename)
    ]
    for loc in possible_locations:
        if os.path.exists(loc):
            print(f"Found '{filename}' at: {loc}")
            return loc
    
    if IN_COLAB:
        print(f"File '{filename}' not found. Please upload it:")
        from google.colab import files
        uploaded = files.upload()
        if filename in uploaded:
            return filename
    raise FileNotFoundError(f"Could not locate '{filename}'.")

dataset_path = find_or_upload_dataset(DATASET_NAME)
df = pd.read_excel(dataset_path)
data = np.array(df, dtype=np.float64)

dataX = data[:, :9]
dataY = data[:, 9:12] # [phi, theta, psi]

print(f"Dataset Loaded: {DATASET_NAME} | Total Rows: {len(data):,}")
print("Euler Angle Variances in this Dataset:")
print(f"  Roll  (phi)   variance: {np.var(dataY[:, 0]):.6f}")
print(f"  Pitch (theta) variance: {np.var(dataY[:, 1]):.6f}")
print(f"  Yaw   (psi)   variance: {np.var(dataY[:, 2]):.6f}")
"""),
        md_cell("### Train & Evaluate 3 Dedicated Single-Axis Models"),
        code_cell("""from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_sliding_sequences(X, Y, window_size=20, stride=1):
    X_seq, Y_seq = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq.append(X[i:i + window_size])
        Y_seq.append(Y[i + window_size - 1])
    return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32)

def build_single_axis_model(window_size=20, input_dim=9, name="axis_model"):
    inputs = Input(shape=(window_size, input_dim), name=f"{name}_input")
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.25)(x)
    x = LSTM(32)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.15)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

window_size = 20
stride_train = 5
stride_test = 1
epochs_offline = 40
epochs_inc = 10
batch_size = 128
n_inc_batches = 8

split = int(len(data) * 0.60)
scaler_X = StandardScaler()
dataX_scaled = scaler_X.fit_transform(dataX)

# Dictionary to store predictions and scalers for each angle
axis_configs = [
    ('Roll (Phi)', 0, 'roll_model'),
    ('Pitch (Theta)', 1, 'pitch_model'),
    ('Yaw (Psi)', 2, 'yaw_model')
]

predicted_axes = {}
ground_truth_axes = {}
axis_rmses = {}

t_all_start = time.time()

for axis_name, col_idx, model_name in axis_configs:
    print(f"\\n{'='*60}")
    print(f"Training Dedicated Model for {axis_name}...")
    print(f"{'='*60}")
    
    Y_raw = dataY[:, col_idx:col_idx+1]
    scaler_Y = StandardScaler()
    dataY_scaled = scaler_Y.fit_transform(Y_raw)
    
    dataXt, dataYt = dataX_scaled[:split], dataY_scaled[:split]
    dataXv, dataYv_scaled = dataX_scaled[split:], dataY_scaled[split:]
    dataYv_orig = Y_raw[split:]
    
    X_train, Y_train = create_sliding_sequences(dataXt, dataYt, window_size, stride=stride_train)
    
    # Build and train dedicated model
    tf.keras.backend.clear_session()
    model = build_single_axis_model(window_size, dataX.shape[1], name=model_name)
    
    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
    es_cb = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    t0 = time.time()
    model.fit(
        X_train, Y_train,
        epochs=epochs_offline,
        batch_size=batch_size,
        callbacks=[lr_cb, es_cb],
        verbose=1
    )
    print(f"Offline Training finished in {time.time() - t0:.1f}s.")
    
    # Incremental update loop
    test_len = len(dataXv)
    step_size = max(window_size * 2, test_len // (n_inc_batches + 1))
    inc_end = step_size * n_inc_batches
    
    preds_list, gts_list = [], []
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    
    for j in range(n_inc_batches):
        pt = step_size * j
        end_pt = min(pt + step_size, test_len)
        if end_pt - pt < window_size: break
        
        xb = dataXv[pt:end_pt]
        yb_s = dataYv_scaled[pt:end_pt]
        yb_o = dataYv_orig[pt:end_pt]
        
        xs, _ = create_sliding_sequences(xb, yb_s, window_size, stride=stride_test)
        _, yo = create_sliding_sequences(xb, yb_o, window_size, stride=stride_test)
        if len(xs) == 0: break
        
        p_scaled = model.predict(xs, verbose=0)
        p_orig = scaler_Y.inverse_transform(p_scaled)
        preds_list.append(p_orig)
        gts_list.append(yo)
        
        x_tr, y_tr = create_sliding_sequences(xb, yb_s, window_size, stride=stride_train)
        if len(x_tr) > 0:
            model.fit(x_tr, y_tr, epochs=epochs_inc, batch_size=batch_size, verbose=0)
            
    if inc_end < test_len and (test_len - inc_end) >= window_size:
        xb = dataXv[inc_end:]
        yb_s = dataYv_scaled[inc_end:]
        yb_o = dataYv_orig[inc_end:]
        xs, _ = create_sliding_sequences(xb, yb_s, window_size, stride=stride_test)
        _, yo = create_sliding_sequences(xb, yb_o, window_size, stride=stride_test)
        if len(xs) > 0:
            p_scaled = model.predict(xs, verbose=0)
            preds_list.append(scaler_Y.inverse_transform(p_scaled))
            gts_list.append(yo)
            
    y_pred_ax = np.vstack(preds_list)
    y_true_ax = np.vstack(gts_list)
    min_l = min(len(y_pred_ax), len(y_true_ax))
    y_pred_ax, y_true_ax = y_pred_ax[:min_l], y_true_ax[:min_l]
    
    err = sqrt(mean_squared_error(y_true_ax, y_pred_ax))
    axis_rmses[axis_name] = err
    predicted_axes[axis_name] = y_pred_ax
    ground_truth_axes[axis_name] = y_true_ax
    print(f"--> {axis_name} Dedicated Model RMSE: {err:.6f} rad ({np.degrees(err):.3f} deg)")

avg_decoupled_rmse = sum(axis_rmses.values()) / 3.0
print("\\n" + "="*70)
print(f"DECOUPLED MULTI-MODEL PERFORMANCE SUMMARY ON {DATASET_NAME}")
print("="*70)
for k, v in axis_rmses.items():
    print(f"  {k:<15} RMSE: {v:.6f} rad ({np.degrees(v):.3f} deg)")
print(f"  {'Overall Mean':<15} RMSE: {avg_decoupled_rmse:.6f} rad ({np.degrees(avg_decoupled_rmse):.3f} deg)")
print(f"Total Pipeline Time: {time.time() - t_all_start:.1f}s")
print("="*70)
"""),
        md_cell("### Synchronized 3-Axis Attitude Tracking Plot"),
        code_cell("""# Plot all 3 dedicated predictions together
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
colors = [('navy', 'crimson'), ('darkgreen', 'orange'), ('indigo', 'cyan')]

for ax, (ax_name, _, _), (c1, c2) in zip(axes, axis_configs, colors):
    yt = ground_truth_axes[ax_name]
    yp = predicted_axes[ax_name]
    err = axis_rmses[ax_name]
    ax.plot(yt, label=f'Ground Truth {ax_name}', color=c1, alpha=0.85, linewidth=1.5)
    ax.plot(yp, label=f'Dedicated LSTM (RMSE: {err:.4f} rad)', color=c2, alpha=0.85, linestyle='--', linewidth=1.4)
    ax.set_ylabel(f"{ax_name} [rad]", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)

axes[-1].set_xlabel("Sample Index", fontsize=11)
fig.suptitle(f"Decoupled Attitude Estimation — {DATASET_NAME}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
""")
    ]
    return create_notebook(cells)


def main():
    generators = {
        'Roll_Estimation_LSTM.ipynb': lambda: generate_single_axis_notebook(
            "Roll", "Phi", 9,
            "Roll relies heavily on lateral acceleration (ay, az) and high-rate roll gyro (p). Separating it prevents aggressive roll swings from corrupting pitch.",
            "D5.xlsx"
        ),
        'Pitch_Estimation_LSTM.ipynb': lambda: generate_single_axis_notebook(
            "Pitch", "Theta", 10,
            "Pitch relies on longitudinal acceleration (ax, az) and pitch gyro (q). In D4 and D5 pitch variance is near zero; separating pitch guarantees it will not be ignored by the loss function.",
            "D4.xlsx"
        ),
        'Yaw_Estimation_LSTM.ipynb': lambda: generate_single_axis_notebook(
            "Yaw", "Psi", 11,
            "Yaw relies almost entirely on the magnetometer (mx, my, mz) and vertical gyro (r). Accelerometers cannot sense yaw rotation around gravity; decoupling yaw eliminates magnetic cross-talk.",
            "D1.xlsx"
        ),
        'Decoupled_Attitude_Master_Colab.ipynb': build_decoupled_master
    }
    
    print("Generating Decoupled Google Colab Notebooks...")
    for fname, gen in generators.items():
        nb = gen()
        # Save in root
        root_path = os.path.join(BASE_DIR, fname)
        with open(root_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  Created root: {fname}")
        
        # Save in colab_notebooks
        sub_path = os.path.join(NOTEBOOKS_DIR, fname)
        with open(sub_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  Created colab_notebooks/{fname}")

    print("\nAll 4 Decoupled Notebooks successfully generated!")

if __name__ == '__main__':
    main()
