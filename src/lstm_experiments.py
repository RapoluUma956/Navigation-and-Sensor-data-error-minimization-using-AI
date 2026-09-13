# -*- coding: utf-8 -*-
"""
LSTM Navigation & Sensor Data Error Minimization — Comprehensive Experiments
=============================================================================
Runs 8 different LSTM approaches on 6 IMU datasets (D1–D6) with incremental learning.
Compiles RMSE results into a comparison table and identifies the best approach.

Datasets: 9 inputs (ax,ay,az,p,q,r,mx,my,mz) → 3 outputs (phi,theta,psi)
"""

import os
import sys
import time
import json
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt

# Suppress TF warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = [f'D{i}.xlsx' for i in range(1, 7)]
RESULTS_DIR = os.path.join(BASE_DIR, 'experiment_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_RATIO = 0.6  # 60% offline training, 40% incremental testing
INCREMENTAL_BATCHES = 8
INPUT_COLS = 9  # ax,ay,az,p,q,r,mx,my,mz
OUTPUT_COLS = 3  # phi,theta,psi


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_dataset(filepath):
    """Load dataset and split into features/targets."""
    df = pd.read_excel(filepath)
    data = np.array(df, dtype=np.float64)
    dataX = data[:, :INPUT_COLS]
    dataY = data[:, INPUT_COLS:INPUT_COLS + OUTPUT_COLS]
    return dataX, dataY, data.shape[0]


def create_sequences_nonoverlap(X, Y, timestep):
    """Non-overlapping reshape (original approach)."""
    n = (len(X) // timestep) * timestep
    X = X[:n]
    Y = Y[:n]
    X_seq = X.reshape(n // timestep, timestep, X.shape[1])
    Y_seq = Y[::timestep]  # Take first target of each window
    return X_seq, Y_seq


def create_sequences_sliding(X, Y, window_size, stride=1):
    """Sliding window with overlap — key improvement for temporal learning."""
    X_seq, Y_seq = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq.append(X[i:i + window_size])
        Y_seq.append(Y[i + window_size - 1])  # Predict the last timestep
    return np.array(X_seq), np.array(Y_seq)


def compute_rmse(y_true, y_pred):
    """Compute RMSE for each of the 3 outputs (phi, theta, psi)."""
    rmse_phi = sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_theta = sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    rmse_psi = sqrt(mean_squared_error(y_true[:, 2], y_pred[:, 2]))
    return rmse_phi, rmse_theta, rmse_psi


def save_prediction_plot(y_true, y_pred, dataset_name, exp_name, save_dir):
    """Save phi/theta/psi prediction plots."""
    labels = ['Phi (Roll)', 'Theta (Pitch)', 'Psi (Yaw)']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(y_true[:, i], label='Reference', alpha=0.8)
        ax.plot(y_pred[:, i], label='Predicted', alpha=0.8)
        ax.set_title(f'{label} — {dataset_name}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Angle (rad)')
        ax.legend()
    plt.suptitle(f'{exp_name} — {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(save_dir, f'{exp_name}_{dataset_name}.png')
    plt.savefig(fname, dpi=100)
    plt.close()


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_experiment_1(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 1: Baseline (Original code — bugs fixed)
    Architecture: LSTM(50) → Dropout(0.25) → LSTM(20) → Dense(3)
    Params: timestep=2, epochs=20, batch=50, RMSprop
    """
    tf.keras.backend.clear_session()
    timestep = 2
    epochs = 20
    batch_size = 50

    # Normalize inputs
    scaler_X = preprocessing.MinMaxScaler()
    dataX = scaler_X.fit_transform(dataX_raw)

    # Dynamic split
    split = int(n_samples * TRAIN_RATIO)
    split = (split // timestep) * timestep  # Ensure divisible

    dataXt = dataX[:split]
    dataYt = dataY[:split]
    dataXv = dataX[split:]
    dataYv = dataY[split:]

    input_dim = dataXt.shape[1]

    # Build sequences for offline training
    X_train, Y_train = create_sequences_nonoverlap(dataXt, dataYt, timestep)

    # Build model
    inputs = Input(shape=[timestep, input_dim])
    layer = LSTM(50, return_sequences=True)(inputs)
    layer = Dropout(0.25)(layer)
    layer = LSTM(20)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='RMSprop', loss='mean_squared_error')

    # Offline training
    model.fit(X_train, Y_train, epochs=epochs, verbose=0, batch_size=batch_size)

    # Incremental learning
    test_len = len(dataXv)
    step_size = max(timestep * 2, (test_len // (INCREMENTAL_BATCHES + 1) // timestep) * timestep)
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []

    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        if pt + step_size > test_len:
            break
        Xbatch = dataXv[pt:pt + step_size]
        Ybatch = dataYv[pt:pt + step_size]

        X_seq, Y_seq = create_sequences_nonoverlap(Xbatch, Ybatch, timestep)

        preds = model.predict(X_seq, verbose=0)
        all_predictions.append(preds)

        # Incremental train
        model.fit(X_seq, Y_seq, epochs=epochs, verbose=0, batch_size=batch_size)

    # Remaining data
    if inc_end < test_len:
        Xrem = dataXv[inc_end:]
        Yrem = dataYv[inc_end:]
        if len(Xrem) >= timestep:
            X_seq, Y_seq = create_sequences_nonoverlap(Xrem, Yrem, timestep)
            preds = model.predict(X_seq, verbose=0)
            all_predictions.append(preds)

    if len(all_predictions) == 0:
        return None, None, None

    results = np.vstack(all_predictions)
    total = results.shape[0]

    # Ground truth (subsample to match)
    Yt = dataYv[::timestep][:total]

    return compute_rmse(Yt, results)


def run_experiment_2(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 2: Hyperparameter Tuning
    Architecture: LSTM(50) → Dropout(0.25) → LSTM(20) → Dense(3)
    Params: timestep=10, epochs=50 offline / 10 inc, batch=128, Adam lr=0.001
    """
    tf.keras.backend.clear_session()
    timestep = 10
    epochs_offline = 50
    epochs_inc = 10
    batch_size = 128

    scaler_X = preprocessing.MinMaxScaler()
    dataX = scaler_X.fit_transform(dataX_raw)

    split = int(n_samples * TRAIN_RATIO)
    split = (split // timestep) * timestep

    dataXt, dataYt = dataX[:split], dataY[:split]
    dataXv, dataYv = dataX[split:], dataY[split:]
    input_dim = dataXt.shape[1]

    X_train, Y_train = create_sequences_nonoverlap(dataXt, dataYt, timestep)

    inputs = Input(shape=[timestep, input_dim])
    layer = LSTM(50, return_sequences=True)(inputs)
    layer = Dropout(0.25)(layer)
    layer = LSTM(20)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Offline with LR reduction
    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    model.fit(X_train, Y_train, epochs=epochs_offline, verbose=0,
              batch_size=batch_size, callbacks=[lr_cb])

    # Incremental learning
    test_len = len(dataXv)
    step_size = max(timestep * 2, (test_len // (INCREMENTAL_BATCHES + 1) // timestep) * timestep)
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []
    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        if pt + step_size > test_len:
            break
        Xbatch = dataXv[pt:pt + step_size]
        Ybatch = dataYv[pt:pt + step_size]
        X_seq, Y_seq = create_sequences_nonoverlap(Xbatch, Ybatch, timestep)
        preds = model.predict(X_seq, verbose=0)
        all_predictions.append(preds)
        model.fit(X_seq, Y_seq, epochs=epochs_inc, verbose=0, batch_size=batch_size)

    if inc_end < test_len:
        Xrem = dataXv[inc_end:]
        Yrem = dataYv[inc_end:]
        if len(Xrem) >= timestep:
            X_seq, Y_seq = create_sequences_nonoverlap(Xrem, Yrem, timestep)
            preds = model.predict(X_seq, verbose=0)
            all_predictions.append(preds)

    if not all_predictions:
        return None, None, None
    results = np.vstack(all_predictions)
    Yt = dataYv[::timestep][:results.shape[0]]
    return compute_rmse(Yt, results)


def run_experiment_3(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 3: Deeper Architecture
    Architecture: LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(3)
    Params: timestep=10, epochs=50 offline / 15 inc, batch=128, Adam
    """
    tf.keras.backend.clear_session()
    timestep = 10
    epochs_offline = 50
    epochs_inc = 15
    batch_size = 128

    scaler_X = preprocessing.MinMaxScaler()
    dataX = scaler_X.fit_transform(dataX_raw)

    split = int(n_samples * TRAIN_RATIO)
    split = (split // timestep) * timestep

    dataXt, dataYt = dataX[:split], dataY[:split]
    dataXv, dataYv = dataX[split:], dataY[split:]
    input_dim = dataXt.shape[1]

    X_train, Y_train = create_sequences_nonoverlap(dataXt, dataYt, timestep)

    inputs = Input(shape=[timestep, input_dim])
    layer = LSTM(128, return_sequences=True)(inputs)
    layer = Dropout(0.3)(layer)
    layer = LSTM(64, return_sequences=True)(layer)
    layer = Dropout(0.2)(layer)
    layer = LSTM(32)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    es_cb = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=epochs_offline, verbose=0,
              batch_size=batch_size, callbacks=[lr_cb, es_cb])

    test_len = len(dataXv)
    step_size = max(timestep * 2, (test_len // (INCREMENTAL_BATCHES + 1) // timestep) * timestep)
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []
    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        if pt + step_size > test_len:
            break
        Xbatch = dataXv[pt:pt + step_size]
        Ybatch = dataYv[pt:pt + step_size]
        X_seq, Y_seq = create_sequences_nonoverlap(Xbatch, Ybatch, timestep)
        preds = model.predict(X_seq, verbose=0)
        all_predictions.append(preds)
        model.fit(X_seq, Y_seq, epochs=epochs_inc, verbose=0, batch_size=batch_size)

    if inc_end < test_len:
        Xrem = dataXv[inc_end:]
        Yrem = dataYv[inc_end:]
        if len(Xrem) >= timestep:
            X_seq, Y_seq = create_sequences_nonoverlap(Xrem, Yrem, timestep)
            preds = model.predict(X_seq, verbose=0)
            all_predictions.append(preds)

    if not all_predictions:
        return None, None, None
    results = np.vstack(all_predictions)
    Yt = dataYv[::timestep][:results.shape[0]]
    return compute_rmse(Yt, results)


def run_experiment_4(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 4: Bidirectional LSTM
    Architecture: BiLSTM(64) → Dropout(0.3) → LSTM(32) → Dense(3)
    Params: timestep=10, epochs=50 offline / 10 inc, batch=128, Adam
    """
    tf.keras.backend.clear_session()
    timestep = 10
    epochs_offline = 50
    epochs_inc = 10
    batch_size = 128

    scaler_X = preprocessing.MinMaxScaler()
    dataX = scaler_X.fit_transform(dataX_raw)

    split = int(n_samples * TRAIN_RATIO)
    split = (split // timestep) * timestep

    dataXt, dataYt = dataX[:split], dataY[:split]
    dataXv, dataYv = dataX[split:], dataY[split:]
    input_dim = dataXt.shape[1]

    X_train, Y_train = create_sequences_nonoverlap(dataXt, dataYt, timestep)

    inputs = Input(shape=[timestep, input_dim])
    layer = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    layer = Dropout(0.3)(layer)
    layer = LSTM(32)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    model.fit(X_train, Y_train, epochs=epochs_offline, verbose=0,
              batch_size=batch_size, callbacks=[lr_cb])

    test_len = len(dataXv)
    step_size = max(timestep * 2, (test_len // (INCREMENTAL_BATCHES + 1) // timestep) * timestep)
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []
    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        if pt + step_size > test_len:
            break
        Xbatch = dataXv[pt:pt + step_size]
        Ybatch = dataYv[pt:pt + step_size]
        X_seq, Y_seq = create_sequences_nonoverlap(Xbatch, Ybatch, timestep)
        preds = model.predict(X_seq, verbose=0)
        all_predictions.append(preds)
        model.fit(X_seq, Y_seq, epochs=epochs_inc, verbose=0, batch_size=batch_size)

    if inc_end < test_len:
        Xrem = dataXv[inc_end:]
        Yrem = dataYv[inc_end:]
        if len(Xrem) >= timestep:
            X_seq, Y_seq = create_sequences_nonoverlap(Xrem, Yrem, timestep)
            preds = model.predict(X_seq, verbose=0)
            all_predictions.append(preds)

    if not all_predictions:
        return None, None, None
    results = np.vstack(all_predictions)
    Yt = dataYv[::timestep][:results.shape[0]]
    return compute_rmse(Yt, results)


def run_experiment_5(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 5: Output Normalization (StandardScaler on both X and Y)
    Architecture: LSTM(50) → Dropout(0.25) → LSTM(20) → Dense(3) (same as Exp2)
    Key change: normalize outputs too, inverse-transform before RMSE
    """
    tf.keras.backend.clear_session()
    timestep = 10
    epochs_offline = 50
    epochs_inc = 10
    batch_size = 128

    scaler_X = preprocessing.StandardScaler()
    scaler_Y = preprocessing.StandardScaler()
    dataX = scaler_X.fit_transform(dataX_raw)
    dataY_norm = scaler_Y.fit_transform(dataY)

    split = int(n_samples * TRAIN_RATIO)
    split = (split // timestep) * timestep

    dataXt, dataYt = dataX[:split], dataY_norm[:split]
    dataXv, dataYv_norm = dataX[split:], dataY_norm[split:]
    dataYv_orig = dataY[split:]  # Keep original for RMSE
    input_dim = dataXt.shape[1]

    X_train, Y_train = create_sequences_nonoverlap(dataXt, dataYt, timestep)

    inputs = Input(shape=[timestep, input_dim])
    layer = LSTM(50, return_sequences=True)(inputs)
    layer = Dropout(0.25)(layer)
    layer = LSTM(20)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    model.fit(X_train, Y_train, epochs=epochs_offline, verbose=0,
              batch_size=batch_size, callbacks=[lr_cb])

    test_len = len(dataXv)
    step_size = max(timestep * 2, (test_len // (INCREMENTAL_BATCHES + 1) // timestep) * timestep)
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []
    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        if pt + step_size > test_len:
            break
        Xbatch = dataXv[pt:pt + step_size]
        Ybatch = dataYv_norm[pt:pt + step_size]
        X_seq, Y_seq = create_sequences_nonoverlap(Xbatch, Ybatch, timestep)
        preds = model.predict(X_seq, verbose=0)
        all_predictions.append(preds)
        model.fit(X_seq, Y_seq, epochs=epochs_inc, verbose=0, batch_size=batch_size)

    if inc_end < test_len:
        Xrem = dataXv[inc_end:]
        Yrem = dataYv_norm[inc_end:]
        if len(Xrem) >= timestep:
            X_seq, Y_seq = create_sequences_nonoverlap(Xrem, Yrem, timestep)
            preds = model.predict(X_seq, verbose=0)
            all_predictions.append(preds)

    if not all_predictions:
        return None, None, None

    results = np.vstack(all_predictions)
    # Inverse-transform predictions back to original scale
    results_orig = scaler_Y.inverse_transform(results)
    Yt = dataYv_orig[::timestep][:results.shape[0]]
    return compute_rmse(Yt, results_orig)


def run_experiment_6(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 6: Sliding Window / Rotational Training
    Architecture: LSTM(64) → Dropout(0.3) → LSTM(32) → Dense(3)
    Key change: overlapping sliding windows (window=20, stride=5)
    """
    tf.keras.backend.clear_session()
    window_size = 20
    stride_train = 5
    stride_test = 1  # Dense predictions at test time
    epochs_offline = 50
    epochs_inc = 10
    batch_size = 128

    scaler_X = preprocessing.MinMaxScaler()
    dataX = scaler_X.fit_transform(dataX_raw)

    split = int(n_samples * TRAIN_RATIO)

    dataXt, dataYt = dataX[:split], dataY[:split]
    dataXv, dataYv = dataX[split:], dataY[split:]
    input_dim = dataXt.shape[1]

    # Sliding window for training
    X_train, Y_train = create_sequences_sliding(dataXt, dataYt, window_size, stride_train)

    inputs = Input(shape=[window_size, input_dim])
    layer = LSTM(64, return_sequences=True)(inputs)
    layer = Dropout(0.3)(layer)
    layer = LSTM(32)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    es_cb = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=epochs_offline, verbose=0,
              batch_size=batch_size, callbacks=[lr_cb, es_cb])

    # Incremental with sliding window
    test_len = len(dataXv)
    step_size = max(window_size * 2, test_len // (INCREMENTAL_BATCHES + 1))
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []
    all_gt = []
    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        end_pt = min(pt + step_size, test_len)
        if end_pt - pt < window_size:
            break
        Xbatch = dataXv[pt:end_pt]
        Ybatch = dataYv[pt:end_pt]
        X_seq, Y_seq = create_sequences_sliding(Xbatch, Ybatch, window_size, stride_test)
        if len(X_seq) == 0:
            break
        preds = model.predict(X_seq, verbose=0)
        all_predictions.append(preds)
        all_gt.append(Y_seq)

        # Incremental training with stride
        X_seq_train, Y_seq_train = create_sequences_sliding(Xbatch, Ybatch, window_size, stride_train)
        if len(X_seq_train) > 0:
            model.fit(X_seq_train, Y_seq_train, epochs=epochs_inc, verbose=0, batch_size=batch_size)

    # Remaining data
    if inc_end < test_len and test_len - inc_end >= window_size:
        Xrem = dataXv[inc_end:]
        Yrem = dataYv[inc_end:]
        X_seq, Y_seq = create_sequences_sliding(Xrem, Yrem, window_size, stride_test)
        if len(X_seq) > 0:
            preds = model.predict(X_seq, verbose=0)
            all_predictions.append(preds)
            all_gt.append(Y_seq)

    if not all_predictions:
        return None, None, None
    results = np.vstack(all_predictions)
    Yt = np.vstack(all_gt)
    # Ensure matching lengths
    min_len = min(len(results), len(Yt))
    return compute_rmse(Yt[:min_len], results[:min_len])


def run_experiment_7(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 7: Combined Best Practices
    Architecture: BiLSTM(128) → Dropout(0.3) → LSTM(64) → Dense(3)
    Params: sliding window=20/stride=5, StandardScaler on X+Y, Adam+LR+ES
    epochs=100 offline / 15 inc, batch=256
    """
    tf.keras.backend.clear_session()
    window_size = 20
    stride_train = 5
    stride_test = 1
    epochs_offline = 100
    epochs_inc = 15
    batch_size = 256

    scaler_X = preprocessing.StandardScaler()
    scaler_Y = preprocessing.StandardScaler()
    dataX = scaler_X.fit_transform(dataX_raw)
    dataY_norm = scaler_Y.fit_transform(dataY)

    split = int(n_samples * TRAIN_RATIO)

    dataXt, dataYt = dataX[:split], dataY_norm[:split]
    dataXv, dataYv_norm = dataX[split:], dataY_norm[split:]
    dataYv_orig = dataY[split:]
    input_dim = dataXt.shape[1]

    X_train, Y_train = create_sequences_sliding(dataXt, dataYt, window_size, stride_train)

    inputs = Input(shape=[window_size, input_dim])
    layer = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    layer = Dropout(0.3)(layer)
    layer = LSTM(64)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    es_cb = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=epochs_offline, verbose=0,
              batch_size=batch_size, callbacks=[lr_cb, es_cb])

    test_len = len(dataXv)
    step_size = max(window_size * 2, test_len // (INCREMENTAL_BATCHES + 1))
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []
    all_gt = []
    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        end_pt = min(pt + step_size, test_len)
        if end_pt - pt < window_size:
            break
        Xbatch = dataXv[pt:end_pt]
        Ybatch_norm = dataYv_norm[pt:end_pt]
        Ybatch_orig = dataYv_orig[pt:end_pt]
        X_seq, Y_seq_norm = create_sequences_sliding(Xbatch, Ybatch_norm, window_size, stride_test)
        _, Y_seq_orig = create_sequences_sliding(Xbatch, Ybatch_orig, window_size, stride_test)
        if len(X_seq) == 0:
            break

        preds_norm = model.predict(X_seq, verbose=0)
        preds_orig = scaler_Y.inverse_transform(preds_norm)
        all_predictions.append(preds_orig)
        all_gt.append(Y_seq_orig)

        X_seq_train, Y_seq_train = create_sequences_sliding(Xbatch, Ybatch_norm, window_size, stride_train)
        if len(X_seq_train) > 0:
            model.fit(X_seq_train, Y_seq_train, epochs=epochs_inc, verbose=0, batch_size=batch_size)

    if inc_end < test_len and test_len - inc_end >= window_size:
        Xrem = dataXv[inc_end:]
        Yrem_norm = dataYv_norm[inc_end:]
        Yrem_orig = dataYv_orig[inc_end:]
        X_seq, Y_seq_norm = create_sequences_sliding(Xrem, Yrem_norm, window_size, stride_test)
        _, Y_seq_orig = create_sequences_sliding(Xrem, Yrem_orig, window_size, stride_test)
        if len(X_seq) > 0:
            preds_norm = model.predict(X_seq, verbose=0)
            preds_orig = scaler_Y.inverse_transform(preds_norm)
            all_predictions.append(preds_orig)
            all_gt.append(Y_seq_orig)

    if not all_predictions:
        return None, None, None
    results = np.vstack(all_predictions)
    Yt = np.vstack(all_gt)
    min_len = min(len(results), len(Yt))
    return compute_rmse(Yt[:min_len], results[:min_len])


def run_experiment_8(dataX_raw, dataY, n_samples, dataset_name):
    """
    Experiment 8: Stacked Bi-LSTM with Dense Attention-like Layers
    Architecture: BiLSTM(128) → Dropout(0.3) → BiLSTM(64) → Dense(64,relu) → Dropout(0.2) → Dense(3)
    Params: sliding window=20/stride=5, StandardScaler X+Y, Adam cosine decay
    epochs=100 offline / 15 inc, batch=256
    """
    tf.keras.backend.clear_session()
    window_size = 20
    stride_train = 5
    stride_test = 1
    epochs_offline = 100
    epochs_inc = 15
    batch_size = 256

    scaler_X = preprocessing.StandardScaler()
    scaler_Y = preprocessing.StandardScaler()
    dataX = scaler_X.fit_transform(dataX_raw)
    dataY_norm = scaler_Y.fit_transform(dataY)

    split = int(n_samples * TRAIN_RATIO)

    dataXt, dataYt = dataX[:split], dataY_norm[:split]
    dataXv, dataYv_norm = dataX[split:], dataY_norm[split:]
    dataYv_orig = dataY[split:]
    input_dim = dataXt.shape[1]

    X_train, Y_train = create_sequences_sliding(dataXt, dataYt, window_size, stride_train)

    # Cosine decay schedule
    total_steps = (len(X_train) // batch_size) * epochs_offline
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=total_steps,
        alpha=1e-6
    )

    inputs = Input(shape=[window_size, input_dim])
    layer = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    layer = Dropout(0.3)(layer)
    layer = Bidirectional(LSTM(64))(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dropout(0.2)(layer)
    outputs = Dense(3, activation='linear')(layer)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error')

    es_cb = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=epochs_offline, verbose=0,
              batch_size=batch_size, callbacks=[es_cb])

    test_len = len(dataXv)
    step_size = max(window_size * 2, test_len // (INCREMENTAL_BATCHES + 1))
    inc_end = step_size * INCREMENTAL_BATCHES

    all_predictions = []
    all_gt = []
    for j in range(INCREMENTAL_BATCHES):
        pt = step_size * j
        end_pt = min(pt + step_size, test_len)
        if end_pt - pt < window_size:
            break
        Xbatch = dataXv[pt:end_pt]
        Ybatch_norm = dataYv_norm[pt:end_pt]
        Ybatch_orig = dataYv_orig[pt:end_pt]
        X_seq, Y_seq_norm = create_sequences_sliding(Xbatch, Ybatch_norm, window_size, stride_test)
        _, Y_seq_orig = create_sequences_sliding(Xbatch, Ybatch_orig, window_size, stride_test)
        if len(X_seq) == 0:
            break

        preds_norm = model.predict(X_seq, verbose=0)
        preds_orig = scaler_Y.inverse_transform(preds_norm)
        all_predictions.append(preds_orig)
        all_gt.append(Y_seq_orig)

        # Re-compile with fixed lr for incremental
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        X_seq_train, Y_seq_train = create_sequences_sliding(Xbatch, Ybatch_norm, window_size, stride_train)
        if len(X_seq_train) > 0:
            model.fit(X_seq_train, Y_seq_train, epochs=epochs_inc, verbose=0, batch_size=batch_size)

    if inc_end < test_len and test_len - inc_end >= window_size:
        Xrem = dataXv[inc_end:]
        Yrem_norm = dataYv_norm[inc_end:]
        Yrem_orig = dataYv_orig[inc_end:]
        X_seq, Y_seq_norm = create_sequences_sliding(Xrem, Yrem_norm, window_size, stride_test)
        _, Y_seq_orig = create_sequences_sliding(Xrem, Yrem_orig, window_size, stride_test)
        if len(X_seq) > 0:
            preds_norm = model.predict(X_seq, verbose=0)
            preds_orig = scaler_Y.inverse_transform(preds_norm)
            all_predictions.append(preds_orig)
            all_gt.append(Y_seq_orig)

    if not all_predictions:
        return None, None, None
    results = np.vstack(all_predictions)
    Yt = np.vstack(all_gt)
    min_len = min(len(results), len(Yt))
    return compute_rmse(Yt[:min_len], results[:min_len])


# ============================================================================
# MAIN: RUN ALL EXPERIMENTS
# ============================================================================

EXPERIMENTS = {
    'Exp1_Baseline_Fixed': run_experiment_1,
    'Exp2_Hyperparam_Tuned': run_experiment_2,
    'Exp3_Deeper_Arch': run_experiment_3,
    'Exp4_Bidirectional': run_experiment_4,
    'Exp5_Output_Normalization': run_experiment_5,
    'Exp6_Sliding_Window': run_experiment_6,
    'Exp7_Combined_Best': run_experiment_7,
    'Exp8_BiLSTM_Dense': run_experiment_8,
}


def main():
    print("=" * 80)
    print("LSTM Navigation Error Minimization — Comprehensive Experiments")
    print("=" * 80)

    # Results storage — resume from existing results if available
    all_results = []
    completed_keys = set()
    results_csv = os.path.join(RESULTS_DIR, 'results_all.csv')
    if os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
        df_existing = df_existing.drop_duplicates(subset=['Dataset', 'Experiment'], keep='last')
        all_results = df_existing.to_dict(orient='records')
        for r in all_results:
            if r.get('Status') == 'OK':
                completed_keys.add((r['Dataset'], r['Experiment']))
        print(f"Resuming: {len(completed_keys)} experiment-dataset pairs already completed.")

    for ds_name in DATASETS:
        ds_path = os.path.join(BASE_DIR, ds_name)
        print(f"\n{'='*60}")
        print(f"Loading dataset: {ds_name}")
        print(f"{'='*60}")

        dataX_raw, dataY, n_samples = load_dataset(ds_path)
        print(f"  Samples: {n_samples}, Input features: {dataX_raw.shape[1]}, Outputs: {dataY.shape[1]}")

        for exp_name, exp_func in EXPERIMENTS.items():
            if (ds_name, exp_name) in completed_keys:
                print(f"\n  SKIPPING {exp_name} on {ds_name} (already completed)", flush=True)
                continue
            print(f"\n  Running {exp_name} on {ds_name}...", flush=True)
            start_time = time.time()

            try:
                rmse_phi, rmse_theta, rmse_psi = exp_func(dataX_raw, dataY, n_samples, ds_name)
                elapsed = time.time() - start_time

                if rmse_phi is not None:
                    avg_rmse = (rmse_phi + rmse_theta + rmse_psi) / 3.0
                    result = {
                        'Dataset': ds_name,
                        'Experiment': exp_name,
                        'RMSE_Phi': round(rmse_phi, 6),
                        'RMSE_Theta': round(rmse_theta, 6),
                        'RMSE_Psi': round(rmse_psi, 6),
                        'Avg_RMSE': round(avg_rmse, 6),
                        'Time_sec': round(elapsed, 1),
                        'Status': 'OK'
                    }
                    print(f"    RMSE: phi={rmse_phi:.6f}, theta={rmse_theta:.6f}, psi={rmse_psi:.6f} "
                          f"(avg={avg_rmse:.6f}) [{elapsed:.1f}s]")
                else:
                    result = {
                        'Dataset': ds_name, 'Experiment': exp_name,
                        'RMSE_Phi': None, 'RMSE_Theta': None, 'RMSE_Psi': None,
                        'Avg_RMSE': None, 'Time_sec': round(elapsed, 1), 'Status': 'SKIP'
                    }
                    print(f"    SKIPPED (insufficient data) [{elapsed:.1f}s]")
            except Exception as e:
                elapsed = time.time() - start_time
                result = {
                    'Dataset': ds_name, 'Experiment': exp_name,
                    'RMSE_Phi': None, 'RMSE_Theta': None, 'RMSE_Psi': None,
                    'Avg_RMSE': None, 'Time_sec': round(elapsed, 1), 'Status': f'ERROR: {str(e)}'
                }
                print(f"    ERROR: {e} [{elapsed:.1f}s]")
                traceback.print_exc()

            all_results.append(result)

            # Save intermediate results after each experiment
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(os.path.join(RESULTS_DIR, 'results_all.csv'), index=False)

    # ================================================================
    # Final Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'results_all.csv'), index=False)

    # Print full table
    print("\n--- Complete Results ---")
    print(df_results.to_string(index=False))

    # Average RMSE per experiment across all datasets
    print("\n--- Average RMSE per Experiment (across all datasets) ---")
    df_ok = df_results[df_results['Status'] == 'OK']
    if len(df_ok) > 0:
        summary = df_ok.groupby('Experiment')[['RMSE_Phi', 'RMSE_Theta', 'RMSE_Psi', 'Avg_RMSE']].mean()
        summary = summary.sort_values('Avg_RMSE')
        print(summary.to_string())

        # Best experiment
        best_exp = summary.index[0]
        best_rmse = summary.loc[best_exp, 'Avg_RMSE']
        print(f"\n*** BEST EXPERIMENT: {best_exp} with average RMSE = {best_rmse:.6f} ***")

        # Save summary
        summary.to_csv(os.path.join(RESULTS_DIR, 'summary_by_experiment.csv'))

        # Best per dataset
        print("\n--- Best Experiment per Dataset ---")
        for ds in DATASETS:
            ds_data = df_ok[df_ok['Dataset'] == ds]
            if len(ds_data) > 0:
                best_row = ds_data.loc[ds_data['Avg_RMSE'].idxmin()]
                print(f"  {ds}: {best_row['Experiment']} (avg RMSE = {best_row['Avg_RMSE']:.6f})")

    # Save results as JSON too
    with open(os.path.join(RESULTS_DIR, 'results_all.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
