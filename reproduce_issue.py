
import os
import shutil
import sys
from unittest.mock import MagicMock
# MOCK MATPLOTLIB TO AVOID ENV ERROR
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.colors"] = MagicMock()

import numpy as np
from cl_benchmark.cl_train import run, DEFAULT_CFG

def run_test():
    # Setup test config
    cfg = DEFAULT_CFG.copy()
    cfg["EPOCHS_PER_TASK"] = 1
    cfg["BATCH_SIZE"] = 16
    cfg["OUTDIR_ROOT"] = "results_test/repro"
    cfg["MVT_ROOT"] = "data/mvtec" # Assuming data exists
    cfg["MVT_LOCO_ROOT"] = "data/mvtec-loco"
    
    # We need to limit tasks, but run() has hardcoded task loading.
    # We can rely on it running all tasks, but we can kill it early? 
    # Or just let it run 2 tasks? 
    # cl_train.run doesn't support task limit arg easily without modifying code.
    # However, we can patch load_mvtec_all_categories?
    
    # Actually, let's just run it as is. It might take too long for all tasks.
    # We can import and modify the task list in the run function? No.
    # PRO TIP: The user provided task list handling in run(). 
    # Let's trust it runs reasonably fast if we set epochs to 1?
    # Actually 20 tasks is a lot.
    
    # Let's Mock the loader to return only 2 categories.
    pass

if __name__ == "__main__":
    import cl_benchmark.cl_train as cl_module
    
    # PATCH LOADERS to return small subest
    original_mvtec = cl_module.load_mvtec_all_categories
    
    def mocked_mvtec(root):
        classes, train, test = original_mvtec(root)
        # return only first 2
        keep = classes[:2]
        new_train = {k: train[k] for k in keep}
        new_test = {k: test[k] for k in keep}
        return keep, new_train, new_test
        
    def mocked_loco(root):
        return [], {}, {} # skip loco
        
    cl_module.load_mvtec_all_categories = mocked_mvtec
    cl_module.load_mvtec_loco_all_categories = mocked_loco
    
    print(">>> Running Finetune...", flush=True)
    cfg_ft = DEFAULT_CFG.copy()
    cfg_ft["CL_METHOD"] = "Finetune"
    cfg_ft["EPOCHS_PER_TASK"] = 1
    cfg_ft["OUTDIR_ROOT"] = "results_test/repro"
    cfg_ft["SEED"] = 42
    acc_ft, auc_ft = cl_module.run(cfg_ft)
    
    print("\n\n>>> Running EWC...", flush=True)
    cfg_ewc = DEFAULT_CFG.copy()
    cfg_ewc["CL_METHOD"] = "EWC"
    cfg_ewc["EPOCHS_PER_TASK"] = 1
    cfg_ewc["OUTDIR_ROOT"] = "results_test/repro"
    cfg_ewc["SEED"] = 42
    acc_ewc, auc_ewc = cl_module.run(cfg_ewc)
    
    print(f"\n\nFinetune ACC: {acc_ft}")
    print(f"EWC ACC: {acc_ewc}")
    
    if acc_ft == acc_ewc:
        print("FAIL: Results are IDENTICAL.")
    else:
        print("SUCCESS: Results are different.")
