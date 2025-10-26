import argparse
import yaml
import torch
import random
import numpy as np
import re

# ---------------------------------------------------------
# 1. Namespace Helper (to access config keys as attributes)
# ---------------------------------------------------------
class Namespace(object):
    """Recursively converts dictionaries into accessible namespaces."""
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match(r"[A-Za-z_-]", key), \
                f"Invalid key name in config: {key}"
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(
            f"Cannot find '{attribute}' in namespace. "
            f"Please include it in your config YAML file!"
        )

# ---------------------------------------------------------
# 2. Deterministic Seed Setup
# ---------------------------------------------------------
def set_deterministic(seed: int):
    """Ensure deterministic results by fixing all random seeds."""
    if seed is not None:
        print(f"[Seed] Setting deterministic mode with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------
# 3. Boolean Converter
# ---------------------------------------------------------
def str2bool(v):
    """Converts common string inputs to boolean."""
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")

# ---------------------------------------------------------
# 4. Argument Parser
# ---------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Continual Anomaly Detection Benchmark")

    # --- Base Paths ---
    parser.add_argument('--config-file', default='./configs/test.yaml', type=str,
                        help='Path to YAML config file (e.g., ./configs/replay.yaml)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--data_dir', type=str, default='./data/mvtec_ad',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--mtd_dir', type=str, default='./data/mtd_ano_mask',
                        help='Path to preprocessed MTD anomaly mask folder (if any)')

    # --- General Training Settings ---
    parser.add_argument('--save_checkpoint', type=str2bool, default=False,
                        help='Whether to save checkpoints')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='Ratio of label noise (if used)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # --- Parse known arguments first ---
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 5. Load YAML Config and Merge with CLI Arguments
    # ---------------------------------------------------------
    try:
        with open(args.config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
            print(f"[Config] Loaded configuration from: {args.config_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    # Convert YAML dict to Namespace for nested access
    config_namespace = Namespace(yaml_data)

    # Merge YAML values into argparse Namespace
    for key, value in config_namespace.__dict__.items():
        vars(args)[key] = value

    # Set deterministic behavior
    set_deterministic(args.seed)

    print(f"[Device] Using device: {args.device}")
    return args


if __name__ == "__main__":
    # Quick test for standalone runs
    args = get_args()
    print("\n=== Loaded Arguments ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
