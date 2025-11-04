"""
datasets package initializer
----------------------------------
This project contains multiple dataset loaders.

For the continual anomaly benchmark, we use:
    get_mvtec_tasks()  --> defined in mvtec_loader.py
"""

# Import only our clean benchmark loader
from .mvtec_loader import get_mvtec_tasks
