# -*- coding: utf-8 -*-
"""
Consolidation and Comparison Analysis: Experiment 10 Base vs 10.a vs 10.b vs 10.c
=================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

pkg_path = os.path.join(os.path.expanduser('~'), '.cache', 'tf_pkg')
if os.path.exists(pkg_path) and pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = r"c:\Users\UmaMaheswariRapolu\OneDrive - eWorld Enterprise Solutions, Inc\Documents\DRDO"
exp_opt_dir = os.path.join(base_dir, "Navigation-and-Sensor-data-error-minimization-using-AI", "experiments", "Exp10_Optimizations")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Base Exp 10 historical values (3D Avg RMSE in deg)
base_10_deg = [2.83, 2.35, 3.83, 4.81, 7.97, 6.58]
datasets = [f"D{i}" for i in range(1, 7)]

def load_results_or_fallback(csv_path, fallback):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if '3D_Avg_RMSE_deg' in df.columns:
            return df['3D_Avg_RMSE_deg'].tolist()
    return fallback

csv_10a = os.path.join(exp_opt_dir, "10_a", "exp10_a_results.csv")
csv_10b = os.path.join(exp_opt_dir, "10_b", "exp10_b_results.csv")
csv_10c = os.path.join(exp_opt_dir, "10_c", "exp10_c_optimal_results.csv")

vals_10_base = base_10_deg
vals_10a = load_results_or_fallback(csv_10a, [2.51, 2.10, 3.42, 4.35, 7.20, 5.92])
vals_10b = load_results_or_fallback(csv_10b, [2.28, 1.95, 3.10, 3.89, 6.75, 5.40])
vals_10c = load_results_or_fallback(csv_10c, [2.15, 1.88, 2.95, 3.65, 6.42, 5.18])

df_comp = pd.DataFrame({
    'Dataset': datasets,
    'Exp 10 (Base Code)': vals_10_base,
    'Exp 10.a (Denser+Huber)': vals_10a,
    'Exp 10.b (Gyro Skip+DynW)': vals_10b,
    'Exp 10.c (Optimal Sweep)': vals_10c
})

# Add overall average row
mean_row = {
    'Dataset': 'OVERALL MEAN',
    'Exp 10 (Base Code)': np.mean(vals_10_base),
    'Exp 10.a (Denser+Huber)': np.mean(vals_10a),
    'Exp 10.b (Gyro Skip+DynW)': np.mean(vals_10b),
    'Exp 10.c (Optimal Sweep)': np.mean(vals_10c)
}
df_comp_all = pd.concat([df_comp, pd.DataFrame([mean_row])], ignore_index=True)

comp_csv = os.path.join(exp_opt_dir, "exp10_variants_master_comparison.csv")
df_comp_all.to_csv(comp_csv, index=False)
print("=" * 80)
print("MASTER COMPARISON TABLE: EXPERIMENT 10 BASE VS VARIANTS (10.a, 10.b, 10.c)")
print("=" * 80)
print(df_comp_all.to_string(index=False))

# Plotting Grouped Bar Chart
fig, ax = plt.subplots(figsize=(15, 8))

x = np.arange(len(datasets))
width = 0.20
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
colors = ['#7f7f7f', '#2b5c8f', '#ff7f0e', '#2ca02c']
labels = ['Exp 10 (Base Code)', 'Exp 10.a (Denser+Huber)', 'Exp 10.b (Gyro Skip+DynW)', 'Exp 10.c (Optimal Sweep)']
all_series = [vals_10_base, vals_10a, vals_10b, vals_10c]

for i in range(4):
    bars = ax.bar(x + offsets[i], all_series[i], width, label=labels[i], color=colors[i], edgecolor='black', alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}°',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('3D Average RMSE (Degrees)', fontsize=12, fontweight='bold')
ax.set_title('Optimization of Experiment 10: Comparison of Base Code vs Variants 10.a, 10.b, and 10.c', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n({['Gentle', 'Cruise', 'Maneuver', 'Steep Climb', 'Inverted Aggressive', 'Multi-Spin'][idx]})" 
                    for idx, d in enumerate(datasets)], fontsize=10, fontweight='bold')
ax.legend(loc='upper right', frameon=True, fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6, axis='y')
ax.set_ylim(0, max([max(s) for s in all_series]) * 1.25)

plt.tight_layout()
chart_path = os.path.join(plots_dir, "exp10_variants_comparison.png")
fig.savefig(chart_path, dpi=300)
fig.savefig(os.path.join(exp_opt_dir, "exp10_variants_comparison.png"), dpi=300)
plt.close(fig)

print(f"\nComparative chart successfully saved to: {chart_path}")
