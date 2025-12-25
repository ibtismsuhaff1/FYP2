import argparse
import yaml
import torch
import random
import numpy as np
import re
import os


# Recursive Namespace Helper
class Namespace(object):
    """Recursively converts dictionaries into accessible namespaces."""

    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match(
                r"[A-Za-z_-]", key
            ), f"Invalid key name in config: {key}"
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(
            f"Cannot find '{attribute}' in namespace. "
            f"Please include it in your config YAML file!"
        )


# Deterministic Seed
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


# String → Boolean Helper
def str2bool(v):
    """Converts common string inputs to boolean."""
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


# Main Argument Parser
def get_args():
    parser = argparse.ArgumentParser(
        description="Continual Anomaly Detection Benchmark"
    )

    # Core paths and devices
    parser.add_argument(
        "--config-file",
        default="./configs/cad.yaml",
        type=str,
        help="Path to YAML config file (e.g., ./configs/cad.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)",
    )

    # Model / checkpoints / misc
    parser.add_argument(
        "--save_checkpoint",
        type=str2bool,
        default=False,
        help="Whether to save checkpoints after training",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load YAML config
    try:
        with open(args.config_file, "r") as f:
            yaml_data = yaml.safe_load(f)
            print(f"[Config] Loaded configuration from: {args.config_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    # Convert YAML dict to Namespace for nested access
    config_namespace = Namespace(yaml_data)

    # Merge YAML data into argparse args
    for key, value in config_namespace.__dict__.items():
        vars(args)[key] = value

    # Handle multiple datasets
    if not hasattr(args, "datasets"):
        raise ValueError("Your YAML config must include a 'datasets:' list.")

    # Validate all datasets
    print("\n[Dataset Configuration]")
    for ds in args.datasets:
        name = ds.get("name", "unknown")
        path = ds.get("data_dir", "./data")
        n_tasks = ds.get("n_tasks", "N/A")

        if not os.path.exists(path):
            print(f"  Warning: Dataset path not found for '{name}' → {path}")
        else:
            print(f"  {name} → {path} ({n_tasks} tasks)")

    # Set deterministic behavior and display info
    set_deterministic(args.seed)
    print(f"\n[Device] Using device: {args.device}")
    print("=" * 60)
    print("Running Continual Anomaly Detection Benchmark")
    print("=" * 60)

    return args


# Entry Point (for quick standalone test)
if __name__ == "__main__":
    args = get_args()
    print("\n=== Loaded Arguments ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
