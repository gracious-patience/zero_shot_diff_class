import torch
import argparse
import logging
from omegaconf import OmegaConf
import numpy as np
import random
import torch
from dally.bert.utils.fs import nirvana
import yaml
from dally.bert.utils.distributed.mpi import mpicomm
from dally.utils import ddp

log = logging.getLogger(__name__)


def print_rank_0(*args):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, flush=True)
    else:
        print(*args, flush=True)


def print_rank(message):
    """print only on any rank."""
    if torch.distributed.is_initialized():
        print(f"rank {torch.distributed.get_rank()}: {message}", flush=True)
    else:
        print(message, flush=True)


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local_rank", required=False, default=0)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if nirvana.ndl:
        if ddp.is_chief():
            log.info(
                "TRAINING IN NIRVANA LOADING CONFIG FROM JSON INPUT \n !! --config has not effect !! \n"
            )
            with open("./configs/nirvana.yaml", "w") as yml:
                yaml.dump(nirvana.params(), yml)
            config = OmegaConf.load("./configs/nirvana.yaml")
        else:
            config = None
        config = mpicomm.bcast(config, root=0)
    else:
        config = OmegaConf.load(args.config)

    return config, args


def prepare_and_fix_seeds(seed=42):
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info(f"Global seed set to {seed}")
