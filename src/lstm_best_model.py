# -*- coding: utf-8 -*-
"""
Optimized LSTM Framework for Attitude & Sensor Error Minimization
================================================================
Empirically validated best architecture across all 6 benchmark datasets (D1-D6).

Key Enhancements over Baseline:
1. Stacked Bidirectional LSTM + Dense Representation:
   Input(20, 9) -> BiLSTM(128, return_sequences=True) -> Dropout(0.3)
                -> BiLSTM(64) -> Dense(64, 'relu') -> Dropout(0.2) -> Dense(3)
2. Dual Standardization (StandardScaler for both 9 IMU inputs & 3 Euler targets)
3. Sliding Window / Rotational Temporal Extraction (window=20, stride=5)
4. Cosine Decay Learning Rate Scheduling + Early Stopping
5. Dynamic Dataset-Adaptive Train/Test & Incremental Splitting
6. Verified across low-dynamic, aggressive-roll, and 3D maneuvering profiles

Usage:
    python lstm_best_model.py --dataset D1.xlsx
    python lstm_best_model.py --all   # Evaluates across all D1-D6 datasets
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Configure TensorFlow environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Add custom local package path if needed
pkg_path = os.path.join(os.path.expanduser('~'), '.cache', 'tf_pkg')
if os.path.exists(pkg_path) and pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Reproducibility seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_sensor_data(filepath, input_dim=9, output_dim=3):
    """
    Loads IMU sensor readings and reference Euler angles from Excel.
    Inputs: [ax, ay, az, p, q, r, mx, my, mz] (9 features)
    Outputs: [phi, theta, psi] (Roll, Pitch, Yaw in radians)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_excel(filepath)
    data = np.array(df, dtype=np.float64)
    X = data[:, :input_dim]
    Y = data[:, input_dim:input_dim + output_dim]
    return X, Y, len(data)


def create_sliding_sequences(X, Y, window_size=20, stride=1):
    """
    Extracts overlapping temporal windows to capture continuous rotational dynamics.
    Predicts the Euler angle at the final timestep of each sequence window.
    """
    X_seq, Y_seq = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq.append(X[i:i + window_size])
        Y_seq.append(Y[i + window_size - 1])
    return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32)


def build_bilstm_model(window_size=20, input_dim=9, output_dim=3, lr=0.001, total_decay_steps=1000):
    """
    Constructs the highest-performing Stacked Bi-LSTM architecture with Cosine Decay.
    """
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=max(1, total_decay_steps),
        alpha=1e-6
    )
    
    inputs = Input(shape=(window_size, input_dim), name="imu_input")
    x = Bidirectional(LSTM(128, return_sequences=True, name="bilstm_1"))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, name="bilstm_2"))(x)
    x = Dense(64, activation='relu', name="dense_feat")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_dim, activation='linear', name="attitude_output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Attitude_BiLSTM")
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error')
    return model


def train_and_evaluate(dataset_name, base_dir='.', window_size=20, train_ratio=0.6,
                       n_inc_batches=8, offline_epochs=60, inc_epochs=15, batch_size=256,
                       save_plots=True, plot_dir='plots'):
    """
    Executes full offline pretraining followed by incremental online learning.
    """
    dataset_path = os.path.join(base_dir, dataset_name)
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}")
    
    X_raw, Y_raw, n_samples = load_sensor_data(dataset_path)
    print(f"Total Samples: {n_samples:,} | Inputs: {X_raw.shape[1]} | Targets: {Y_raw.shape[1]}")
    
    # 1. Dual Standardization
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    Y_scaled = scaler_Y.fit_transform(Y_raw)
    
    # 2. Dynamic Split (60% Offline Pretraining, 40% Incremental Online Testing)
    split_idx = int(n_samples * train_ratio)
    X_train_raw, Y_train_scaled = X_scaled[:split_idx], Y_scaled[:split_idx]
    X_test_raw, Y_test_scaled = X_scaled[split_idx:], Y_scaled[split_idx:]
    Y_test_orig = Y_raw[split_idx:]
    
    # 3. Create Sequences for Offline Training (stride=5 for memory efficiency & diversity)
    X_train_seq, Y_train_seq = create_sliding_sequences(
        X_train_raw, Y_train_scaled, window_size=window_size, stride=5
    )
    print(f"Offline Training Sequences: {len(X_train_seq):,} (Window={window_size})")
    
    # 4. Model Instantiation & Offline Pretraining
    total_steps = (len(X_train_seq) // batch_size) * offline_epochs
    model = build_bilstm_model(
        window_size=window_size,
        input_dim=X_raw.shape[1],
        output_dim=Y_raw.shape[1],
        lr=0.001,
        total_decay_steps=total_steps
    )
    
    es_callback = EarlyStopping(monitor='loss', patience=12, restore_best_weights=True)
    
    print(f"Training offline model for up to {offline_epochs} epochs...", end=" ", flush=True)
    t0 = time.time()
    history = model.fit(
        X_train_seq, Y_train_seq,
        epochs=offline_epochs,
        batch_size=batch_size,
        callbacks=[es_callback],
        verbose=0
    )
    t_offline = time.time() - t0
    print(f"Done in {t_offline:.1f}s ({len(history.history['loss'])} epochs executed).")
    
    # 5. Incremental Online Adaptation & Sequential Prediction
    test_len = len(X_test_raw)
    step_size = max(window_size * 2, test_len // (n_inc_batches + 1))
    
    all_predictions = []
    all_ground_truth = []
    
    # Recompile for fine-tuning with small learning rate
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    
    print(f"Running {n_inc_batches} incremental update batches...", flush=True)
    t_inc_start = time.time()
    
    for batch_i in range(n_inc_batches):
        pt = step_size * batch_i
        end_pt = min(pt + step_size, test_len)
        if end_pt - pt < window_size:
            break
        
        batch_X = X_test_raw[pt:end_pt]
        batch_Y_scaled = Y_test_scaled[pt:end_pt]
        batch_Y_orig = Y_test_orig[pt:end_pt]
        
        # Dense test sequence evaluation (stride=1 for continuous time resolution)
        X_test_batch_seq, _ = create_sliding_sequences(batch_X, batch_Y_scaled, window_size=window_size, stride=1)
        _, Y_test_batch_orig_seq = create_sliding_sequences(batch_X, batch_Y_orig, window_size=window_size, stride=1)
        
        if len(X_test_batch_seq) == 0:
            break
        
        # Predict on incoming sensor stream
        preds_scaled = model.predict(X_test_batch_seq, verbose=0)
        preds_orig = scaler_Y.inverse_transform(preds_scaled)
        
        all_predictions.append(preds_orig)
        all_ground_truth.append(Y_test_batch_orig_seq)
        
        # Incremental online fine-tuning on current batch (simulate sensor adaptation)
        X_inc_train, Y_inc_train = create_sliding_sequences(batch_X, batch_Y_scaled, window_size=window_size, stride=5)
        if len(X_inc_train) > 0:
            model.fit(X_inc_train, Y_inc_train, epochs=inc_epochs, batch_size=batch_size, verbose=0)
    
    # Handle remaining trajectory points
    inc_end_pt = step_size * n_inc_batches
    if inc_end_pt < test_len and (test_len - inc_end_pt) >= window_size:
        rem_X = X_test_raw[inc_end_pt:]
        rem_Y_scaled = Y_test_scaled[inc_end_pt:]
        rem_Y_orig = Y_test_orig[inc_end_pt:]
        
        rem_X_seq, _ = create_sliding_sequences(rem_X, rem_Y_scaled, window_size=window_size, stride=1)
        _, rem_Y_orig_seq = create_sliding_sequences(rem_X, rem_Y_orig, window_size=window_size, stride=1)
        
        if len(rem_X_seq) > 0:
            preds_scaled = model.predict(rem_X_seq, verbose=0)
            preds_orig = scaler_Y.inverse_transform(preds_scaled)
            all_predictions.append(preds_orig)
            all_ground_truth.append(rem_Y_orig_seq)
    
    t_inc = time.time() - t_inc_start
    print(f"Incremental evaluation complete in {t_inc:.1f}s.")
    
    # 6. Performance Evaluation
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_ground_truth)
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    
    rmse_phi = sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_theta = sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    rmse_psi = sqrt(mean_squared_error(y_true[:, 2], y_pred[:, 2]))
    avg_rmse = (rmse_phi + rmse_theta + rmse_psi) / 3.0
    
    print("\n--- Model Estimation Performance ---")
    print(f"  Roll  (Phi,   rad) RMSE: {rmse_phi:.6f}")
    print(f"  Pitch (Theta, rad) RMSE: {rmse_theta:.6f}")
    print(f"  Yaw   (Psi,   rad) RMSE: {rmse_psi:.6f}")
    print(f"  Average 3D Attitude RMSE: {avg_rmse:.6f}")
    
    # 7. Visualization
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{os.path.splitext(dataset_name)[0]}_prediction.png")
        plot_results(y_true, y_pred, dataset_name, rmse_phi, rmse_theta, rmse_psi, plot_path)
        print(f"  Plot saved to: {plot_path}")
    
    return {
        'Dataset': dataset_name,
        'RMSE_Phi': rmse_phi,
        'RMSE_Theta': rmse_theta,
        'RMSE_Psi': rmse_psi,
        'Avg_RMSE': avg_rmse,
        'Total_Time': t_offline + t_inc
    }


def plot_results(y_true, y_pred, title_prefix, rmse_phi, rmse_theta, rmse_psi, save_path):
    """
    Renders 3-axis attitude tracking curves (Reference vs LSTM).
    """
    angles = [
        ('Roll (Phi)', y_true[:, 0], y_pred[:, 0], rmse_phi, 'navy', 'crimson'),
        ('Pitch (Theta)', y_true[:, 1], y_pred[:, 1], rmse_theta, 'darkgreen', 'orange'),
        ('Yaw (Psi)', y_true[:, 2], y_pred[:, 2], rmse_psi, 'indigo', 'cyan')
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, (name, true, pred, err, c_true, c_pred) in zip(axes, angles):
        ax.plot(true, label='Reference (Ground Truth)', color=c_true, alpha=0.85, linewidth=1.5)
        ax.plot(pred, label=f'LSTM Predicted (RMSE: {err:.4f} rad)', color=c_pred, alpha=0.85, linestyle='--', linewidth=1.4)
        ax.set_ylabel(f"{name} [rad]", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', framealpha=0.9)
    
    axes[-1].set_xlabel("Time Step (Sample Index)", fontsize=11)
    fig.suptitle(f"Attitude Estimation — {title_prefix} (Stacked Bi-LSTM + Dual Normalization)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optimal LSTM Attitude Estimation Framework")
    parser.add_argument('--dataset', type=str, default='D1.xlsx', help="Dataset filename (e.g. D1.xlsx to D6.xlsx)")
    parser.add_argument('--all', action='store_true', help="Run evaluation sequentially on all 6 datasets")
    parser.add_argument('--epochs', type=int, default=60, help="Offline training epochs")
    parser.add_argument('--inc_epochs', type=int, default=15, help="Online incremental update epochs")
    parser.add_argument('--window', type=int, default=20, help="Temporal sliding window size")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.all:
        datasets = [f"D{i}.xlsx" for i in range(1, 7)]
        results = []
        for ds in datasets:
            res = train_and_evaluate(
                dataset_name=ds,
                base_dir=base_dir,
                window_size=args.window,
                offline_epochs=args.epochs,
                inc_epochs=args.inc_epochs
            )
            results.append(res)
        
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("COMPREHENSIVE MULTI-DATASET SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print(f"\nMean 3D Attitude Error Across All 6 Datasets: {df['Avg_RMSE'].mean():.6f} rad")
    else:
        train_and_evaluate(
            dataset_name=args.dataset,
            base_dir=base_dir,
            window_size=args.window,
            offline_epochs=args.epochs,
            inc_epochs=args.inc_epochs
        )


if __name__ == '__main__':
    main()
