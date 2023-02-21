import datetime
import logging
import os

try:
    import torch
    import torch.distributed as dist
except ImportError:
    # Something is completely wrong
    # You are in big troubles
    dist = None

try:
    import deepspeed
except ImportError:
    deepspeed = None

from dally.bert.utils.distributed.mpi import mpicomm
from dally.bert.utils.distributed.constants import backend_var_name, deepspeed_name


def _initialize():
    if dist is None:
        # we cannot rise exception as it brokes compilation
        print("torch.distributed is not imported")
        return
    is_mpirun = not (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
    )
    if is_mpirun:
        from mpi4py import MPI
        import subprocess

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        master_addr = None
        if rank == 0:
            hostname_cmd = ["hostname -I"]
            result = subprocess.check_output(hostname_cmd, shell=True)
            master_addr = result.decode("utf-8").split()[0]
        master_addr = comm.bcast(master_addr, root=0)

        # Determine local rank by assuming hostnames are unique
        proc_name = MPI.Get_processor_name()
        all_procs = comm.allgather(proc_name)
        local_rank = sum([i == proc_name for i in all_procs[:rank]])
        uniq_proc_names = set(all_procs)
        host_rank = sorted(uniq_proc_names).index(proc_name)
        # it is used by deepspeed to determine local rank
        if os.environ[backend_var_name] == deepspeed_name:
            import sys

            sys.argv += ["--local_rank", str(local_rank)]
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["HOST_RANK"] = str(host_rank)
        os.environ["NUM_HOSTS"] = str(len(uniq_proc_names))

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = os.environ.get(
            "MASTER_PORT", "29500"
        )  # TORCH_DISTRIBUTED_DEFAULT_PORT
        os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    # Initialize torch distributed
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(0, 1800))


def local_host_gather(data):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    host_rank = os.environ["HOST_RANK"]
    all_data = comm.allgather((host_rank, data))
    return [d[1] for d in all_data if d[0] == host_rank]


_initialize()


def in_distributed_mode():
    return dist is not None


def is_chief():
    return worker_idx() == 0


def is_local_chief():
    return worker_local_idx() == 0


def worker_idx():
    return dist.get_rank() if in_distributed_mode() else 0


def worker_local_idx():
    return int(os.environ["LOCAL_RANK"])


def worker_host_idx():
    return int(os.environ["HOST_RANK"])


def num_hosts():
    return int()


def num_workers():
    return dist.get_world_size() if in_distributed_mode() else 1


def gpu_visible_device_list():
    return str(dist.get_rank()) if in_distributed_mode() else None


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DummyWork:
    def is_completed(self):
        return True

    def wait(self):
        pass


class CallableNone:
    def __init__(self, async_op_idx=-1, async_op_name="async_op"):
        self._async_op_idx = async_op_idx
        self._async_op_name = async_op_name

    def __call__(self, *args, **kwargs):
        if (
            len(args) > self._async_op_idx
            and args[self._async_op_idx]
            or self._async_op_name in kwargs
            and kwargs[self._async_op_name]
        ):
            return DummyWork()
        return None


allreduce = dist.all_reduce if in_distributed_mode() else CallableNone(3, "async_op")
allgather = dist.all_gather if in_distributed_mode() else CallableNone(3, "async_op")
barrier = dist.barrier if in_distributed_mode() else CallableNone()
reduce = dist.reduce if in_distributed_mode() else CallableNone(4, "async_op")


def broadcast_cpu(data, src, group=None):
    if not in_distributed_mode():
        return data

    # This can fix errors on older versions of torch
    # if dist is not None:
    #     group = dist.group.WORLD

    length = [len(data) if data is not None else 0]
    dist.broadcast_object_list(length, src=src, group=group)
    if worker_idx() != src:
        data = [None] * length[0]
    dist.broadcast_object_list(data, src=src, group=group)
    return data


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Logger:
    def __init__(self, name=__name__, cuda=False):
        self.logger = logging.getLogger(name)
        self.cuda = cuda

    def info(self, message, *args, **kwargs):
        if (self.cuda and dist.get_rank() == 0) or not self.cuda:
            self.logger.info(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        if (self.cuda and dist.get_rank() == 0) or not self.cuda:
            self.logger.debug(message, *args, **kwargs)
