# -*- coding: utf-8 -*-
"""
Experiment 10.b: Decoupled Single-Axis with Gyro Kinematic Skip & Dynamic Windowing
===================================================================================
Optimizations over Exp 10.a:
  1. Dynamic Axis-Specific Windowing:
     - Roll: W = 15 (reduces phase lag during rapid banking)
     - Pitch: W = 15 (fast leveling response)
     - Yaw: W = 25 (extended temporal receptive field for compass/drift filtering)
  2. Direct Gyro Kinematic Skip Connection:
     - Linear bypass highway from instantaneous gyro rate (p, q, r) directly to output.
     - Handles direct angular rate tracking, freeing recurrent LSTM for bias and non-linearities.
  3. Denser Stride (2), Huber Loss (0.1), and LeakyReLU (0.1).
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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, LeakyReLU, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def build_axis_model_10b(window_size=20, input_dim=9, gyro_idx=3, name="model_10b"):
    inputs = Input(shape=(window_size, input_dim), name=f"{name}_in")
    
    # Recurrent LSTM Highway
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dense(32)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(0.1)(x)
    lstm_out = Dense(1, activation='linear', name=f"{name}_lstm_out")(x)
    
    # Gyro Kinematic Skip Connection (Latest instantaneous gyro rate)
    gyro_rate = Lambda(lambda t: t[:, -1, gyro_idx:gyro_idx+1], name=f"{name}_gyro_slice")(inputs)
    gyro_skip = Dense(1, use_bias=False, name=f"{name}_gyro_proj")(gyro_rate)
    
    # Additive Fusion
    outputs = Add(name=f"{name}_out")([lstm_out, gyro_skip])
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=0.1))
    return model


def train_single_axis_10b(X_raw, Y_target_angle, axis_name, axis_idx,
                           train_ratio=0.6, epochs_offline=30, epochs_inc=6, 
                           batch_size=128, n_inc_batches=8):
    # Dynamic Window Sizing
    # Roll: 15, Pitch: 15, Yaw: 25
    if 'roll' in axis_name.lower():
        window_size = 15
        gyro_idx = 3 # p
    elif 'pitch' in axis_name.lower():
        window_size = 15
        gyro_idx = 4 # q
    else:
        window_size = 25
        gyro_idx = 5 # r

    raw_target_1d = Y_target_angle.squeeze()
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

    stride_train = 2
    stride_test = 1

    X_train, Y_train = create_sliding_sequences(dataXt, dataYt, window_size, stride=stride_train)

    tf.keras.backend.clear_session()
    clean_name = axis_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    model = build_axis_model_10b(window_size, X_raw.shape[1], gyro_idx=gyro_idx, name=f"m10b_{clean_name}")

    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
    es_cb = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)

    t0 = time.time()
    model.fit(
        X_train, Y_train,
        epochs=epochs_offline,
        batch_size=batch_size,
        callbacks=[lr_cb, es_cb],
        verbose=0
    )
    t_off = time.time() - t0

    test_len = len(dataXv)
    step_size = max(window_size * 2, test_len // (n_inc_batches + 1))
    inc_end = step_size * n_inc_batches

    preds_list, gts_raw_list = [], []
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=Huber(delta=0.1))

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

        preds_unw = model.predict(xs, verbose=0) * std_Y + mean_Y
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
            preds_list.append(model.predict(xs, verbose=0) * std_Y + mean_Y)
            gts_raw_list.append(yo_raw)

    y_pred_unw = np.vstack(preds_list)
    y_true_raw = np.vstack(gts_raw_list).squeeze()
    min_l = min(len(y_pred_unw), len(y_true_raw))
    y_pred_unw = y_pred_unw[:min_l].squeeze()
    y_true_raw = y_true_raw[:min_l]

    y_pred_wrapped = ((y_pred_unw + np.pi) % (2 * np.pi)) - np.pi
    ang_diff = np.arctan2(np.sin(y_pred_wrapped - y_true_raw), np.cos(y_pred_wrapped - y_true_raw))
    ang_rmse = sqrt(np.mean(ang_diff ** 2))

    return {
        'true_angle': y_true_raw,
        'pred_angle': y_true_raw + ang_diff,
        'rmse_rad': ang_rmse,
        'rmse_deg': np.degrees(ang_rmse),
        'max_err_deg': np.max(np.degrees(np.abs(ang_diff))),
        'window_size': window_size,
        'train_time': t_off
    }


def run_experiment_10b():
    base_dir = r"c:\Users\UmaMaheswariRapolu\OneDrive - eWorld Enterprise Solutions, Inc\Documents\DRDO"
    output_dir = os.path.join(base_dir, "Navigation-and-Sensor-data-error-minimization-using-AI", 
                              "experiments", "Exp10_Optimizations", "10_b")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("EXPERIMENT 10.b: Decoupled with Gyro Kinematic Skip & Dynamic Windowing")
    print("=" * 80)
    
    datasets = [f"D{i}" for i in range(1, 7)]
    axes = [('Roll', 0), ('Pitch', 1), ('Yaw', 2)]
    
    all_results = []
    
    for ds_name in datasets:
        file_path = os.path.join(base_dir, f"{ds_name}.xlsx")
        print(f"\nProcessing {ds_name}...")
        X_raw, Y_raw, N = load_clean_dataset(file_path)
        ds_record = {'Dataset': ds_name, 'Samples': N}
        
        for axis_name, axis_idx in axes:
            target_angle = Y_raw[:, axis_idx:axis_idx+1]
            print(f"  Training dedicated {axis_name} model...", end=" ", flush=True)
            res = train_single_axis_10b(X_raw, target_angle, axis_name, axis_idx)
            ds_record[f'{axis_name}_RMSE_rad'] = res['rmse_rad']
            ds_record[f'{axis_name}_RMSE_deg'] = res['rmse_deg']
            ds_record[f'{axis_name}_MaxErr_deg'] = res['max_err_deg']
            print(f"Done! (W={res['window_size']}) RMSE: {res['rmse_rad']:.4f} rad ({res['rmse_deg']:.2f} deg)")
            
        avg_rad = (ds_record['Roll_RMSE_rad'] + ds_record['Pitch_RMSE_rad'] + ds_record['Yaw_RMSE_rad']) / 3.0
        avg_deg = (ds_record['Roll_RMSE_deg'] + ds_record['Pitch_RMSE_deg'] + ds_record['Yaw_RMSE_deg']) / 3.0
        ds_record['3D_Avg_RMSE_rad'] = avg_rad
        ds_record['3D_Avg_RMSE_deg'] = avg_deg
        
        print(f"  --> {ds_name} 3D Average RMSE: {avg_rad:.4f} rad ({avg_deg:.2f} deg)")
        all_results.append(ds_record)
        
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "exp10_b_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")
    print("\nSummary Table:")
    print(df_results[['Dataset', 'Roll_RMSE_deg', 'Pitch_RMSE_deg', 'Yaw_RMSE_deg', '3D_Avg_RMSE_deg']].to_string(index=False))
    return df_results


if __name__ == '__main__':
    run_experiment_10b()
