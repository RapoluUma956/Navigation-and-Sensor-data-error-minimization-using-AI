# -*- coding: utf-8 -*-
"""
Master Orchestration: Executes Experiment 10.a, 10.b, 10.c, and Comparison
"""

import os
import sys
import subprocess
import time

base_dir = r"c:\Users\UmaMaheswariRapolu\OneDrive - eWorld Enterprise Solutions, Inc\Documents\DRDO"
opt_dir = os.path.join(base_dir, "Navigation-and-Sensor-data-error-minimization-using-AI", "experiments", "Exp10_Optimizations")
python_exe = r"C:\Users\UmaMaheswariRapolu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

scripts = [
    ("Experiment 10.a (Denser Stride + Huber + LeakyReLU)", os.path.join(opt_dir, "10_a", "run_exp10_a.py")),
    ("Experiment 10.b (Gyro Skip + Dynamic Windowing)", os.path.join(opt_dir, "10_b", "run_exp10_b.py")),
    ("Experiment 10.c (Hyperparameter Sweep & Optimal)", os.path.join(opt_dir, "10_c", "run_exp10_c.py")),
    ("Master Comparison & Visualization", os.path.join(opt_dir, "compare_exp10_variants.py"))
]

for title, script_path in scripts:
    print("\n" + "#" * 80)
    print(f"STARTING: {title}")
    print(f"Script: {script_path}")
    print("#" * 80 + "\n")
    t0 = time.time()
    res = subprocess.run([python_exe, script_path], cwd=base_dir)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR executing {script_path}! Return code: {res.returncode}")
        sys.exit(res.returncode)
    print(f"\nCOMPLETED: {title} in {elapsed:.1f}s")

print("\nALL OPTIMIZATION VARIANTS (10.a, 10.b, 10.c) SUCCESSFULLY EXECUTED AND COMPARED!")
