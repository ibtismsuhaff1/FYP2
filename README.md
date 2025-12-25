# Continual Anomaly Detection using Vision Transformers and Continual Learning

## Overview
This project implements a **Continual Anomaly Detection (CAD) system** for industrial inspection scenarios, where anomaly detection models are trained sequentially across multiple tasks without catastrophic forgetting.

The system integrates:
- Vision-based feature extractors (ResNet18 / Vision Transformer)
- Distribution Normalization Embedding (DNE) for anomaly scoring
- Continual Learning (CL) strategies
- MVTec AD + MVTec-LOCO datasets

A complete benchmarking and visualization pipeline is provided.

---

## Objectives
1. Perform anomaly detection in a continual learning setting  
2. Mitigate catastrophic forgetting  
3. Integrate DNE-based anomaly memory  
4. Support multiple CL strategies  
5. Log task-wise accuracy and AUC  
6. Visualize benchmarking results using a GUI  

---

## Datasets
### MVTec AD
Hazelnut, Zipper, Screw, Leather, Transistor, Capsule, Tile, Bottle, Carpet

### MVTec-LOCO
Splicing connectors, Breakfast box, Screw bag, Pushpins, Juice bottle

Expected structure:
```
data/
├── mvtec/
└── mvtec-loco/
```

---

## Methodology
- Backbone: ResNet18 (pretrained) or Vision Transformer
- DNE for anomaly memory
- Sequential task training
- Continual Learning strategies: Finetune, Replay, EWC, LwF, GPM

---

## Evaluation Metrics
- AUC per task
- Task-wise Accuracy
- Accuracy & AUC matrices
- Forgetting analysis

---

## Project Structure
```
cl_benchmark/
├── agents/
├── loaders/
├── utils/
├── cl_train.py
├── test.py
├── config.yaml

gui/
└── app.py

results/
└── mvtec+loco/CL/
```

---

## Training
```
python -m cl_benchmark.cl_train --config cl_benchmark/config.yaml
```

## Evaluation
```
python test.py --mem_dir results/mvtec+loco/CL/finetune_resnet18_pretrained
```

---

## GUI Benchmarking
```
streamlit run gui/app.py
```

---

## Requirements
See requirements.txt

---

## Author
Numaan Suhaff
