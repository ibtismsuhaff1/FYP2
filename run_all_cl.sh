#!/bin/bash

METHODS=(
  "finetune"
  "replay"
  "ewc"
  "lwf"
  "gpm"
)

BASE_DIR="results/mvtec+loco/CL"

for METHOD in "${METHODS[@]}"; do

  EXP_DIR="$BASE_DIR/${METHOD}_resnet18_pretrained"

  echo "============================================================"
  echo "TRAINING METHOD: $METHOD"
  echo "============================================================"

  # Remove old results
  rm -rf "$EXP_DIR"

  # Train
  python -m cl_benchmark.cl_train --set CL_METHOD=$METHOD

  echo "------------------------------------------------------------"
  echo "RUNNING TEST FOR: $METHOD"
  echo "------------------------------------------------------------"

  python test.py --mem_dir "$EXP_DIR"

  echo "DONE WITH: $METHOD"
  echo ""
done

echo "ALL 5 METHODS COMPLETED!"
