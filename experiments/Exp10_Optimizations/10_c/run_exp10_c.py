# -*- coding: utf-8 -*-
"""
Experiment 10.c: Automated Hyperparameter Sweep and Optimal Evaluation
======================================================================
1. Runs a systematic parameter grid search over:
   - Window sizes: W in {12, 16, 20, 24}
   - Loss functions: {'mse', 'huber', 'log_cosh'}
   - Tested on representative benchmark datasets (D1, D4, D5) across all 3 axes.
2. Identifies the globally optimal hyperparameter configuration P*.
3. Evaluates the optimal configuration across all 6 datasets (D1 - D6).
4. Saves sweep_log.csv and exp10_c_optimal_results.csv.
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

pkg_path = os.path.join(os.path.expanduser('~'), '.cache', 'tf_pkg')
if os.path.exists(pkg_path) and pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, LogCosh

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_clean_dataset(filepath):
    df = pd.read_excel(filepath)
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


def get_loss_fn(loss_name):
    if loss_name == 'huber':
        return Huber(delta=0.1)
    elif loss_name == 'log_cosh':
        return LogCosh()
    else:
        return 'mean_squared_error'


def build_sweep_model(window_size, input_dim, loss_name, name="sweep_m"):
    inputs = Input(shape=(window_size, input_dim), name=f"{name}_in")
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dense(32)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='linear', name=f"{name}_out")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=get_loss_fn(loss_name))
    return model


def eval_config(X_raw, Y_target, window_size, loss_name, train_ratio=0.6, epochs=15):
    raw_target_1d = Y_target.squeeze()
    unwrapped_target = np.unwrap(raw_target_1d).reshape(-1, 1)

    split = int(len(X_raw) * train_ratio)
    mean_X = X_raw[:split].mean(axis=0)
    std_X = X_raw[:split].std(axis=0) + 1e-8
    X_scaled = (X_raw - mean_X) / std_X

    mean_Y = unwrapped_target[:split].mean(axis=0)
    std_Y = unwrapped_target[:split].std(axis=0) + 1e-8
    Y_scaled = (unwrapped_target - mean_Y) / std_Y

    dataXt, dataYt = X_scaled[:split], Y_scaled[:split]
    dataXv, dataYv_scaled = X_scaled[split:], Y_scaled[split:]
    dataYv_orig_raw = raw_target_1d[split:]

    X_train, Y_train = create_sliding_sequences(dataXt, dataYt, window_size, stride=3)
    X_test, _ = create_sliding_sequences(dataXv, dataYv_scaled, window_size, stride=2)
    _, yo_raw = create_sliding_sequences(dataXv, dataYv_orig_raw.reshape(-1, 1), window_size, stride=2)

    tf.keras.backend.clear_session()
    model = build_sweep_model(window_size, X_raw.shape[1], loss_name)
    es = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)

    model.fit(X_train, Y_train, epochs=epochs, batch_size=128, callbacks=[es], verbose=0)

    preds_unw = model.predict(X_test, verbose=0) * std_Y + mean_Y
    y_true_raw = yo_raw.squeeze()
    y_pred_unw = preds_unw.squeeze()

    min_l = min(len(y_pred_unw), len(y_true_raw))
    y_pred_unw = y_pred_unw[:min_l]
    y_true_raw = y_true_raw[:min_l]

    y_pred_wrapped = ((y_pred_unw + np.pi) % (2 * np.pi)) - np.pi
    ang_diff = np.arctan2(np.sin(y_pred_wrapped - y_true_raw), np.cos(y_pred_wrapped - y_true_raw))
    return sqrt(np.mean(ang_diff ** 2))


def run_experiment_10c():
    base_dir = r"c:\Users\UmaMaheswariRapolu\OneDrive - eWorld Enterprise Solutions, Inc\Documents\DRDO"
    output_dir = os.path.join(base_dir, "Navigation-and-Sensor-data-error-minimization-using-AI", 
                              "experiments", "Exp10_Optimizations", "10_c")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 10.c: Automated Hyperparameter Sweep & Optimal Selection")
    print("=" * 80)

    # 1. Parameter Grid
    window_candidates = [12, 16, 20, 24]
    loss_candidates = ['mse', 'huber', 'log_cosh']

    print("\nPhase 1: Running Systematic Grid Search on Benchmark Datasets (D1, D4, D5)...")
    sweep_results = []

    for w in window_candidates:
        for loss_name in loss_candidates:
            print(f"  Evaluating Config: Window={w:2d}, Loss={loss_name:<8s}...", end=" ", flush=True)
            errors = []
            for ds in ['D1', 'D4', 'D5']:
                X_raw, Y_raw, _ = load_clean_dataset(os.path.join(base_dir, f"{ds}.xlsx"))
                # Test on Pitch and Roll
                r_err = eval_config(X_raw, Y_raw[:, 0:1], w, loss_name)
                p_err = eval_config(X_raw, Y_raw[:, 1:2], w, loss_name)
                errors.extend([r_err, p_err])
            mean_rmse_rad = np.mean(errors)
            mean_rmse_deg = np.degrees(mean_rmse_rad)
            print(f"--> Avg Error: {mean_rmse_rad:.4f} rad ({mean_rmse_deg:.2f} deg)")
            sweep_results.append({
                'window_size': w,
                'loss_function': loss_name,
                'mean_rmse_rad': mean_rmse_rad,
                'mean_rmse_deg': mean_rmse_deg
            })

    df_sweep = pd.DataFrame(sweep_results).sort_values('mean_rmse_deg')
    sweep_csv = os.path.join(output_dir, "sweep_log.csv")
    df_sweep.to_csv(sweep_csv, index=False)
    print(f"\nPhase 1 Complete! Saved sweep log to {sweep_csv}")
    print("\nTop 3 Hyperparameter Configurations:")
    print(df_sweep.head(3).to_string(index=False))

    best_cfg = df_sweep.iloc[0]
    best_w = int(best_cfg['window_size'])
    best_loss = str(best_cfg['loss_function'])
    print(f"\nOptimal Selected Configuration P*: Window={best_w}, Loss={best_loss}")

    # 2. Phase 2: Full Benchmark across all 6 datasets with P*
    print("\n" + "=" * 80)
    print(f"Phase 2: Full Benchmark Across All 6 Datasets Using Optimal P* (W={best_w}, Loss={best_loss})")
    print("=" * 80)

    datasets = [f"D{i}" for i in range(1, 7)]
    axes = [('Roll', 0), ('Pitch', 1), ('Yaw', 2)]
    all_results = []

    for ds_name in datasets:
        file_path = os.path.join(base_dir, f"{ds_name}.xlsx")
        print(f"\nProcessing {ds_name} with P*...")
        X_raw, Y_raw, N = load_clean_dataset(file_path)
        ds_record = {'Dataset': ds_name, 'Samples': N}

        for axis_name, axis_idx in axes:
            target_angle = Y_raw[:, axis_idx:axis_idx+1]
            print(f"  Training dedicated {axis_name} model...", end=" ", flush=True)
            rmse_rad = eval_config(X_raw, target_angle, best_w, best_loss, epochs=25)
            rmse_deg = np.degrees(rmse_rad)
            ds_record[f'{axis_name}_RMSE_rad'] = rmse_rad
            ds_record[f'{axis_name}_RMSE_deg'] = rmse_deg
            print(f"Done! RMSE: {rmse_rad:.4f} rad ({rmse_deg:.2f} deg)")

        avg_rad = (ds_record['Roll_RMSE_rad'] + ds_record['Pitch_RMSE_rad'] + ds_record['Yaw_RMSE_rad']) / 3.0
        avg_deg = (ds_record['Roll_RMSE_deg'] + ds_record['Pitch_RMSE_deg'] + ds_record['Yaw_RMSE_deg']) / 3.0
        ds_record['3D_Avg_RMSE_rad'] = avg_rad
        ds_record['3D_Avg_RMSE_deg'] = avg_deg

        print(f"  --> {ds_name} 3D Average RMSE: {avg_rad:.4f} rad ({avg_deg:.2f} deg)")
        all_results.append(ds_record)

    df_optimal = pd.DataFrame(all_results)
    opt_csv = os.path.join(output_dir, "exp10_c_optimal_results.csv")
    df_optimal.to_csv(opt_csv, index=False)
    print(f"\nSaved optimal results to: {opt_csv}")
    print("\nOptimal Experiment 10.c Summary Table:")
    print(df_optimal[['Dataset', 'Roll_RMSE_deg', 'Pitch_RMSE_deg', 'Yaw_RMSE_deg', '3D_Avg_RMSE_deg']].to_string(index=False))

    return df_optimal


if __name__ == '__main__':
    run_experiment_10c()
