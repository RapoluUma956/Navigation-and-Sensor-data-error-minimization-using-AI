# -*- coding: utf-8 -*-
"""
Experiment 11: Decoupled S^1 / SO(2) Continuous Embedding with Full Kinematic Cross-Coupling
=============================================================================================
Addresses high dynamic attitude estimation errors on D4, D5, and D6 by:
  1. Continuous SO(2) Unit-Circle Embedding: Models predict [sin(theta), cos(theta)] pairs.
     Eliminates both +-pi boundary step-discontinuities and multi-revolution unbounded drift.
  2. Coupling-Aware Inputs: Each decoupled sub-network receives full 9-axis IMU features
     [ax, ay, az, p, q, r, mx, my, mz] to resolve steep-climb (theta ~ 80 deg) gyro cross-coupling.
  3. Geodesic Reconstruction: Final angle recovered via atan2(sin_hat, cos_hat).
"""

import os
import sys
import time
from math import sqrt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure tf_pkg is in sys.path
pkg_path = os.path.join(os.path.expanduser('~'), '.cache', 'tf_pkg')
if os.path.exists(pkg_path) and pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_clean_dataset(filepath):
    df = pd.read_excel(filepath)
    # Strip uncalibrated initial rows (e.g. rows 0-2 in D4, D5, D6)
    if (df.iloc[0:3][['phi', 'theta', 'psi']] == 0).all().all():
        df = df.iloc[3:].reset_index(drop=True)
    data = np.array(df, dtype=np.float64)
    X = data[:, :9]   # ax, ay, az, p, q, r, mx, my, mz
    Y = data[:, 9:12] # phi, theta, psi
    return X, Y, len(data)


def create_sliding_sequences_sincos(X, Y_angle, window_size=20, stride=1):
    sin_y = np.sin(Y_angle)
    cos_y = np.cos(Y_angle)
    Y_sincos = np.hstack([sin_y, cos_y]) # Shape (N, 2)
    
    X_seq, Y_seq, Y_raw_seq = [], [], []
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq.append(X[i:i + window_size])
        Y_seq.append(Y_sincos[i + window_size - 1])
        Y_raw_seq.append(Y_angle[i + window_size - 1])
        
    return (np.array(X_seq, dtype=np.float32), 
            np.array(Y_seq, dtype=np.float32), 
            np.array(Y_raw_seq, dtype=np.float32).squeeze())


def build_sincos_axis_model(window_size=20, input_dim=9, name="sincos_model"):
    inputs = Input(shape=(window_size, input_dim), name=f"{name}_in")
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(2, activation='linear', name=f"{name}_out")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def train_eval_single_axis(X_raw, Y_target_angle, axis_name, window_size=20, 
                           train_ratio=0.6, epochs=30, batch_size=128):
    """
    Trains a dedicated model predicting [sin(theta), cos(theta)] for one axis.
    Reconstructs angle with atan2 and calculates geodesic angular RMSE.
    """
    N = len(X_raw)
    split = int(N * train_ratio)
    
    mean_X = X_raw[:split].mean(axis=0)
    std_X = X_raw[:split].std(axis=0) + 1e-8
    X_scaled = (X_raw - mean_X) / std_X
    
    # Training set (stride=2 for efficiency)
    X_train, Y_train, _ = create_sliding_sequences_sincos(
        X_scaled[:split], Y_target_angle[:split], window_size=window_size, stride=2
    )
    
    # Testing set (stride=1 for continuous evaluation)
    X_test, _, Y_test_raw = create_sliding_sequences_sincos(
        X_scaled[split:], Y_target_angle[split:], window_size=window_size, stride=1
    )
    
    tf.keras.backend.clear_session()
    clean_name = axis_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    model = build_sincos_axis_model(window_size, X_raw.shape[1], name=f"m_{clean_name}")
    
    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
    es_cb = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)
    
    t0 = time.time()
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_cb, es_cb],
        verbose=0
    )
    train_time = time.time() - t0
    
    preds_sincos = model.predict(X_test, verbose=0)
    pred_sin = preds_sincos[:, 0]
    pred_cos = preds_sincos[:, 1]
    
    norm = np.sqrt(pred_sin**2 + pred_cos**2) + 1e-8
    pred_sin /= norm
    pred_cos /= norm
    
    pred_angle = np.arctan2(pred_sin, pred_cos)
    true_angle = Y_test_raw
    
    # True geodesic angular error on S^1
    ang_diff = np.arctan2(np.sin(pred_angle - true_angle), np.cos(pred_angle - true_angle))
    rmse_rad = sqrt(np.mean(ang_diff ** 2))
    rmse_deg = np.degrees(rmse_rad)
    max_err_deg = np.max(np.degrees(np.abs(ang_diff)))
    mean_err_deg = np.mean(np.degrees(np.abs(ang_diff)))
    
    return {
        'true_angle': true_angle,
        'pred_angle': pred_angle,
        'ang_diff': ang_diff,
        'rmse_rad': rmse_rad,
        'rmse_deg': rmse_deg,
        'max_err_deg': max_err_deg,
        'mean_err_deg': mean_err_deg,
        'train_time': train_time
    }


def run_experiment_11():
    base_dir = r"c:\Users\UmaMaheswariRapolu\OneDrive - eWorld Enterprise Solutions, Inc\Documents\DRDO"
    output_dir = os.path.join(base_dir, "Navigation-and-Sensor-data-error-minimization-using-AI", "experiments", "Exp11_Decoupled_SinCos")
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 80)
    print("EXPERIMENT 11: Decoupled S^1 / SO(2) [Sin, Cos] Embedding + Full Kinematic Coupling")
    print("=" * 80)
    
    datasets = [f"D{i}" for i in range(1, 7)]
    axes = [('Roll', 0), ('Pitch', 1), ('Yaw', 2)]
    
    all_results = []
    all_eval_data = {}
    
    for ds_name in datasets:
        file_path = os.path.join(base_dir, f"{ds_name}.xlsx")
        print(f"\nProcessing {ds_name}...")
        X_raw, Y_raw, N = load_clean_dataset(file_path)
        print(f"  Clean samples: {N:,}")
        
        ds_record = {'Dataset': ds_name, 'Samples': N}
        all_eval_data[ds_name] = {}
        
        for axis_name, axis_idx in axes:
            target_angle = Y_raw[:, axis_idx:axis_idx+1]
            print(f"  Training dedicated {axis_name} model...", end=" ", flush=True)
            res = train_eval_single_axis(X_raw, target_angle, axis_name)
            
            ds_record[f'{axis_name}_RMSE_rad'] = res['rmse_rad']
            ds_record[f'{axis_name}_RMSE_deg'] = res['rmse_deg']
            ds_record[f'{axis_name}_MaxErr_deg'] = res['max_err_deg']
            all_eval_data[ds_name][axis_name] = res
            
            print(f"Done! RMSE: {res['rmse_rad']:.4f} rad ({res['rmse_deg']:.2f} deg), Max: {res['max_err_deg']:.2f} deg")
            
        avg_rad = (ds_record['Roll_RMSE_rad'] + ds_record['Pitch_RMSE_rad'] + ds_record['Yaw_RMSE_rad']) / 3.0
        avg_deg = (ds_record['Roll_RMSE_deg'] + ds_record['Pitch_RMSE_deg'] + ds_record['Yaw_RMSE_deg']) / 3.0
        ds_record['3D_Avg_RMSE_rad'] = avg_rad
        ds_record['3D_Avg_RMSE_deg'] = avg_deg
        
        print(f"  --> {ds_name} 3D Average RMSE: {avg_rad:.4f} rad ({avg_deg:.2f} deg)")
        all_results.append(ds_record)
        
    df_results = pd.DataFrame(all_results)
    
    csv_path = os.path.join(output_dir, "exp11_sincos_results.csv")
    df_results.to_csv(csv_path, index=False)
    csv_root = os.path.join(base_dir, "experiment_results", "exp11_sincos_results.csv")
    os.makedirs(os.path.dirname(csv_root), exist_ok=True)
    df_results.to_csv(csv_root, index=False)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 11 SUMMARY TABLE ACROSS ALL DATASETS:")
    print("=" * 80)
    print(df_results[['Dataset', 'Roll_RMSE_deg', 'Pitch_RMSE_deg', 'Yaw_RMSE_deg', '3D_Avg_RMSE_deg']].to_string(index=False))
    
    mean_roll = df_results['Roll_RMSE_deg'].mean()
    mean_pitch = df_results['Pitch_RMSE_deg'].mean()
    mean_yaw = df_results['Yaw_RMSE_deg'].mean()
    mean_3d_deg = df_results['3D_Avg_RMSE_deg'].mean()
    mean_3d_rad = df_results['3D_Avg_RMSE_rad'].mean()
    
    print(f"\nOverall Project Mean across all 6 datasets:")
    print(f"  Roll: {mean_roll:.2f} deg | Pitch: {mean_pitch:.2f} deg | Yaw: {mean_yaw:.2f} deg")
    print(f"  3D Average RMSE: {mean_3d_rad:.4f} rad ({mean_3d_deg:.2f} deg)")
    
    generate_exp11_plots(all_eval_data, output_dir, plots_dir)
    return df_results


def generate_exp11_plots(all_eval_data, output_dir, plots_dir):
    print("\nGenerating Experiment 11 Comparison Plots...")
    
    for ds_name in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        data = all_eval_data[ds_name]
        fig, axes_pl = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        colors = {'Roll': '#1f77b4', 'Pitch': '#ff7f0e', 'Yaw': '#2ca02c'}
        
        for idx, (axis_name, color) in enumerate(colors.items()):
            ax = axes_pl[idx]
            res = data[axis_name]
            time_idx = np.arange(len(res['true_angle']))
            
            true_deg = np.degrees(res['true_angle'])
            pred_deg = np.degrees(res['pred_angle'])
            
            ax.plot(time_idx, true_deg, label=f'Ground Truth {axis_name}', color='black', alpha=0.7, lw=1.5)
            ax.plot(time_idx, pred_deg, label=f'Exp 11 Decoupled Sin/Cos {axis_name}', color=color, alpha=0.85, lw=1.2, linestyle='--')
            
            ax.set_ylabel(f'{axis_name} (deg)', fontsize=11, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', frameon=True)
            ax.set_title(f"{ds_name} - {axis_name} Tracking (RMSE: {res['rmse_deg']:.2f} deg, Max Err: {res['max_err_deg']:.2f} deg)", fontsize=11)
            
        axes_pl[-1].set_xlabel('Testing Sample Index', fontsize=11, fontweight='bold')
        plt.suptitle(f"Experiment 11: Decoupled S^1 [Sin, Cos] SO(2) Embedding on {ds_name}", fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()
        
        plot_name = f"exp11_{ds_name}_sincos_tracking.png"
        fig.savefig(os.path.join(output_dir, plot_name), dpi=300)
        fig.savefig(os.path.join(plots_dir, plot_name), dpi=300)
        plt.close(fig)
        print(f"  Saved plot: {plot_name}")
        
    fig, axes_pl = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes_pl.flatten()
    
    datasets = [f"D{i}" for i in range(1, 7)]
    for i, ds_name in enumerate(datasets):
        ax = axes_flat[i]
        d_roll = all_eval_data[ds_name]['Roll']
        d_pitch = all_eval_data[ds_name]['Pitch']
        d_yaw = all_eval_data[ds_name]['Yaw']
        
        bar_labels = ['Roll', 'Pitch', 'Yaw']
        bar_vals = [d_roll['rmse_deg'], d_pitch['rmse_deg'], d_yaw['rmse_deg']]
        bars = ax.bar(bar_labels, bar_vals, color=['#2b5c8f', '#d95f02', '#238b45'], width=0.55, edgecolor='black', alpha=0.85)
        
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f} deg',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        ax.set_title(f"{ds_name} (3D Avg: {np.mean(bar_vals):.2f} deg)", fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (degrees)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        ax.set_ylim(0, max(bar_vals) * 1.25 + 1.0)
        
    plt.suptitle("Experiment 11: Decoupled SO(2) [Sin, Cos] Attitude Estimation Across All 6 Datasets", 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    overview_plot = "exp11_all_datasets_comparison.png"
    fig.savefig(os.path.join(output_dir, overview_plot), dpi=300)
    fig.savefig(os.path.join(plots_dir, overview_plot), dpi=300)
    plt.close(fig)
    print(f"  Saved overview plot: {overview_plot}")
    print("All plots generated successfully!\n")


if __name__ == '__main__':
    run_experiment_11()
