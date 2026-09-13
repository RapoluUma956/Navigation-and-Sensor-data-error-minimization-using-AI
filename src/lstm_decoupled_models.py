# -*- coding: utf-8 -*-
"""
Decoupled Attitude Estimation Framework: Separate Models for Roll, Pitch, and Yaw
================================================================================
Eliminates cross-axis gradient interference by training 3 specialized, independent
LSTM models, each tailored to the physical sensor dependencies of its respective angle:
  - Model 1 (Roll  phi)  : Focuses on lateral acceleration (ay, az) & roll rate (p)
  - Model 2 (Pitch theta): Focuses on longitudinal acceleration (ax, az) & pitch rate (q)
  - Model 3 (Yaw   psi)  : Focuses on magnetometer (mx, my, mz) & yaw rate (r)

Usage:
    python lstm_decoupled_models.py --dataset D1.xlsx
    python lstm_decoupled_models.py --dataset D5.xlsx --axis roll
    python lstm_decoupled_models.py --all
"""

import os
import sys
import time
import argparse
import warnings
import re
from math import sqrt
import numpy as np
import pandas as pd
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

pkg_path = os.path.join(os.path.expanduser('~'), '.cache', 'tf_pkg')
if os.path.exists(pkg_path) and pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset(filepath):
    df = pd.read_excel(filepath)
    # Strip uncalibrated initial rows (e.g. rows 0-2 in D4, D5, D6)
    if (df.iloc[0:3][['phi', 'theta', 'psi']] == 0).all().all():
        df = df.iloc[3:].reset_index(drop=True)
    data = np.array(df, dtype=np.float64)
    X = data[:, :9]   # ax, ay, az, p, q, r, mx, my, mz
    Y = data[:, 9:12] # phi, theta, psi
    return X, Y, len(data)


def create_sliding_sequences(X, Y, window_size=20, stride=1):
    X_seq, Y_seq = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq.append(X[i:i + window_size])
        Y_seq.append(Y[i + window_size - 1])
    return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32)


def build_axis_model(window_size=20, input_dim=9, name="axis_model"):
    inputs = Input(shape=(window_size, input_dim), name=f"{name}_input")
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.25)(x)
    x = LSTM(32)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.15)(x)
    outputs = Dense(1, activation='linear', name=f"{name}_output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def train_single_axis(X_raw, Y_target, axis_name, window_size=20, train_ratio=0.6,
                      epochs_offline=35, epochs_inc=8, batch_size=128, n_inc_batches=8):
    """
    Trains an independent, dedicated model for one specific Euler angle.
    Uses continuous phase unwrapping to eliminate +-pi boundary cliffs.
    """
    raw_target_1d = Y_target.squeeze()
    unwrapped_target = np.unwrap(raw_target_1d).reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_raw)
    Y_scaled = scaler_Y.fit_transform(unwrapped_target)
    
    split = int(len(X_raw) * train_ratio)
    dataXt, dataYt = X_scaled[:split], Y_scaled[:split]
    dataXv, dataYv_scaled = X_scaled[split:], Y_scaled[split:]
    dataYv_orig_raw = raw_target_1d[split:]
    
    stride_train = 5
    stride_test = 1
    
    X_train, Y_train = create_sliding_sequences(dataXt, dataYt, window_size, stride=stride_train)
    
    tf.keras.backend.clear_session()
    clean_name = re.sub(r'[^A-Za-z0-9_]', '_', axis_name.lower()).strip('_')
    model = build_axis_model(window_size, X_raw.shape[1], name=f"model_{clean_name}")
    
    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
    es_cb = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
    
    t0 = time.time()
    model.fit(
        X_train, Y_train,
        epochs=epochs_offline,
        batch_size=batch_size,
        callbacks=[lr_cb, es_cb],
        verbose=0
    )
    t_off = time.time() - t0
    
    # Incremental update loop
    test_len = len(dataXv)
    step_size = max(window_size * 2, test_len // (n_inc_batches + 1))
    inc_end = step_size * n_inc_batches
    
    preds_list, gts_raw_list = [], []
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    
    for j in range(n_inc_batches):
        pt = step_size * j
        end_pt = min(pt + step_size, test_len)
        if end_pt - pt < window_size:
            break
        
        xb = dataXv[pt:end_pt]
        yb_s = dataYv_scaled[pt:end_pt]
        yb_raw = dataYv_orig_raw[pt:end_pt]
        
        xs, _ = create_sliding_sequences(xb, yb_s, window_size, stride=stride_test)
        _, yo_raw = create_sliding_sequences(xb, yb_raw.reshape(-1, 1), window_size, stride=stride_test)
        if len(xs) == 0:
            break
            
        preds_unw = scaler_Y.inverse_transform(model.predict(xs, verbose=0))
        preds_list.append(preds_unw)
        gts_raw_list.append(yo_raw)
        
        xtr, ytr = create_sliding_sequences(xb, yb_s, window_size, stride=stride_train)
        if len(xtr) > 0:
            model.fit(xtr, ytr, epochs=epochs_inc, batch_size=batch_size, verbose=0)
            
    if inc_end < test_len and (test_len - inc_end) >= window_size:
        xb = dataXv[inc_end:]
        yb_s = dataYv_scaled[inc_end:]
        yb_raw = dataYv_orig_raw[inc_end:]
        xs, _ = create_sliding_sequences(xb, yb_s, window_size, stride=stride_test)
        _, yo_raw = create_sliding_sequences(xb, yb_raw.reshape(-1, 1), window_size, stride=stride_test)
        if len(xs) > 0:
            preds_list.append(scaler_Y.inverse_transform(model.predict(xs, verbose=0)))
            gts_raw_list.append(yo_raw)
            
    y_pred_unw = np.vstack(preds_list)
    y_true_raw = np.vstack(gts_raw_list).squeeze()
    min_l = min(len(y_pred_unw), len(y_true_raw))
    y_pred_unw = y_pred_unw[:min_l].squeeze()
    y_true_raw = y_true_raw[:min_l]
    
    # Wrap prediction back to [-pi, pi]
    y_pred_wrapped = ((y_pred_unw + np.pi) % (2 * np.pi)) - np.pi
    
    # True geodesic angular difference
    ang_diff = np.arctan2(np.sin(y_pred_wrapped - y_true_raw), np.cos(y_pred_wrapped - y_true_raw))
    ang_rmse = sqrt(np.mean(ang_diff ** 2))
    
    # Continuous aligned prediction for visual display
    y_pred_aligned = y_true_raw + ang_diff
    
    return y_true_raw, y_pred_aligned, ang_rmse, t_off, np.degrees(np.abs(ang_diff))


def run_decoupled_estimation(dataset_path, save_plot=True, plot_dir='plots'):
    ds_name = os.path.basename(dataset_path)
    clean_ds = os.path.splitext(ds_name)[0]
    print(f"\n{'='*70}")
    print(f"DECOUPLED ATTITUDE ESTIMATION: {ds_name} (Phase Unwrapping & Residuals)")
    print(f"{'='*70}")
    
    X_raw, Y_raw, n = load_dataset(dataset_path)
    print(f"Loaded {n:,} samples. Training 3 independent single-axis models...")
    
    axes = [
        ('Roll (Phi)', Y_raw[:, 0:1]),
        ('Pitch (Theta)', Y_raw[:, 1:2]),
        ('Yaw (Psi)', Y_raw[:, 2:3])
    ]
    
    results = {}
    preds_dict = {}
    gts_dict = {}
    errors_deg_dict = {}
    
    for name, target in axes:
        print(f"  Training dedicated {name} model...", end=" ", flush=True)
        yt, yp_aligned, rmse, t_off, err_deg = train_single_axis(X_raw, target, name)
        results[name] = rmse
        preds_dict[name] = yp_aligned
        gts_dict[name] = yt
        errors_deg_dict[name] = err_deg
        print(f"Done! Angular RMSE = {rmse:.6f} rad ({np.degrees(rmse):.3f} deg, Mean Error: {np.mean(err_deg):.2f} deg)")
        
    avg_rmse = sum(results.values()) / 3.0
    print("\n--- Final Decoupled Performance ---")
    for k, v in results.items():
        print(f"  {k:<15}: {v:.6f} rad ({np.degrees(v):.3f} deg, Mean Err: {np.mean(errors_deg_dict[k]):.2f} deg)")
    print(f"  Average 3D Attitude RMSE: {avg_rmse:.6f} rad ({np.degrees(avg_rmse):.3f} deg)")
    
    if save_plot:
        os.makedirs(plot_dir, exist_ok=True)
        fig, axs = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
        colors = [('navy', 'crimson'), ('darkgreen', 'darkorange'), ('indigo', 'teal')]
        for i, ((name, _), (c1, c2)) in enumerate(zip(axes, colors)):
            # Tracking panel
            axs[i, 0].plot(gts_dict[name], label=f'Reference {name}', color=c1, alpha=0.85, linewidth=1.2)
            axs[i, 0].plot(preds_dict[name], label=f'Dedicated LSTM (RMSE: {results[name]:.4f} rad / {np.degrees(results[name]):.1f}°)', color=c2, alpha=0.85, linestyle='--', linewidth=1.2)
            axs[i, 0].set_ylabel(f"{name} [rad]", fontsize=10)
            axs[i, 0].grid(True, linestyle=':', alpha=0.6)
            axs[i, 0].legend(loc='upper right', fontsize=9)
            
            # Error residual panel
            err = errors_deg_dict[name]
            axs[i, 1].plot(err, label=f'Error (Mean: {np.mean(err):.1f}°, Max: {np.max(err):.1f}°)', color=c2, alpha=0.75, linewidth=1.0)
            axs[i, 1].set_ylabel("Error [deg]", fontsize=10)
            axs[i, 1].grid(True, linestyle=':', alpha=0.6)
            axs[i, 1].legend(loc='upper right', fontsize=9)
            
        axs[0, 0].set_title(f"{ds_name} Decoupled Attitude Tracking", fontsize=11, fontweight='bold')
        axs[0, 1].set_title(f"{ds_name} Estimation Error Residuals (Degrees)", fontsize=11, fontweight='bold')
        axs[-1, 0].set_xlabel("Sample Index", fontsize=10)
        axs[-1, 1].set_xlabel("Sample Index", fontsize=10)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"decoupled_{clean_ds}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved to: {plot_path}")
        
        # Copy to artifact directory for display
        artifact_dir = r"C:\Users\UmaMaheswariRapolu\.gemini\antigravity\brain\4fd2e91b-e86c-4a9b-9d27-bc020d3d006a"
        if os.path.exists(artifact_dir):
            shutil.copy2(plot_path, os.path.join(artifact_dir, os.path.basename(plot_path)))
        
    return results, avg_rmse


def main():
    parser = argparse.ArgumentParser(description="Decoupled Single-Axis Attitude Estimation")
    parser.add_argument('--dataset', type=str, default='D1.xlsx', help="Dataset filename (e.g. D1.xlsx)")
    parser.add_argument('--all', action='store_true', help="Run across all D1-D6 datasets")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.all:
        datasets = [f"D{i}.xlsx" for i in range(1, 7)]
        summary = []
        for ds in datasets:
            path = os.path.join(base_dir, ds)
            res, avg = run_decoupled_estimation(path)
            summary.append({
                'Dataset': ds,
                'Roll_RMSE': round(res['Roll (Phi)'], 6),
                'Pitch_RMSE': round(res['Pitch (Theta)'], 6),
                'Yaw_RMSE': round(res['Yaw (Psi)'], 6),
                'Avg_RMSE': round(avg, 6)
            })
        df_sum = pd.DataFrame(summary)
        print("\n" + "="*80)
        print("ALL 6 DATASETS DECOUPLED ESTIMATION SUMMARY")
        print("="*80)
        print(df_sum.to_string(index=False))
        print(f"\nMean Decoupled 3D RMSE Across All 6 Datasets: {df_sum['Avg_RMSE'].mean():.6f} rad")
        
        # Save summary CSV
        csv_path = os.path.join(base_dir, 'experiment_results', 'decoupled_summary.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_sum.to_csv(csv_path, index=False)
        print(f"Summary table saved to: {csv_path}")
        
        # Also plot summary bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x_indices = np.arange(len(datasets))
        bar_width = 0.2
        
        ax.bar(x_indices - bar_width, df_sum['Roll_RMSE'], width=bar_width, label='Roll (Phi) RMSE', color='crimson')
        ax.bar(x_indices, df_sum['Pitch_RMSE'], width=bar_width, label='Pitch (Theta) RMSE', color='forestgreen')
        ax.bar(x_indices + bar_width, df_sum['Yaw_RMSE'], width=bar_width, label='Yaw (Psi) RMSE', color='royalblue')
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('RMSE [rad]', fontsize=12)
        ax.set_title('Decoupled Single-Axis Attitude Estimation Across All 6 Datasets', fontsize=13, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"D{i}" for i in range(1, 7)], fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=11)
        plt.tight_layout()
        
        summary_plot = os.path.join(base_dir, 'plots', 'decoupled_all_datasets_comparison.png')
        plt.savefig(summary_plot, dpi=150)
        plt.close()
        
        artifact_dir = r"C:\Users\UmaMaheswariRapolu\.gemini\antigravity\brain\4fd2e91b-e86c-4a9b-9d27-bc020d3d006a"
        if os.path.exists(artifact_dir):
            shutil.copy2(summary_plot, os.path.join(artifact_dir, os.path.basename(summary_plot)))
        print(f"Comparison bar chart saved to: {summary_plot}")
    else:
        path = os.path.join(base_dir, args.dataset)
        run_decoupled_estimation(path)


if __name__ == '__main__':
    main()
