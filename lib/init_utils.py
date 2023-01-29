from config import cfg as base_config
import os
import torch
import random
import numpy as np

def load_config(config_file):
    cfg = base_config.clone()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    return cfg

def create_config(args, print_warning=False):
    cfg = base_config.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #ensure_config_correctness(cfg, print_warning)
    cfg.freeze()

    return cfg

def init_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def set_cudnn_deterministic():
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_philly():
    PIP3 = "/opt/conda/bin/pip"
    setup = []
    setup.append("sudo apt-get update")
    setup.append("sudo apt-get install libyaml-dev")
    setup.append("sudo {} install pyyaml --upgrade --force".format(PIP3))

    for cmd in setup:
        print(cmd)
        os.system(cmd)