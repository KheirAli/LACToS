"""Seed numpy/torch/random, then run the target script as __main__ (so the
random subsampling in evaluate.py's --max_reconstructions is reproducible)."""
import os, sys, random, runpy
import numpy as np, torch
seed = int(os.environ.get("ABL_SEED", "0"))
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.getcwd())
runpy.run_path(script, run_name="__main__")
