# cl_benchmark/run_all.py
import os
from cl_benchmark.cl_benchmark import run, DEFAULT_CFG

# choose datasets and methods to run
DATASETS = ["MNIST", "CIFAR10"]  # pick any two (as lecturer suggested)
METHODS = ["Finetune", "Replay", "GPM", "SGP", "EWC", "LwF"]

# tweak common settings for quicker tests (short-run)
DEFAULT_CFG["EPOCHS_PER_TASK"] = 2
DEFAULT_CFG["BATCH_SIZE"] = 128

for ds in DATASETS:
    for m in METHODS:
        cfg = DEFAULT_CFG.copy()
        cfg["DATASET_NAME"] = ds
        cfg["CL_METHOD"] = m
        print(f"\n\n>>> RUN: dataset={ds} method={m}")
        out = run(cfg)
        print(f"Done: {ds} - {m}")
