import torch
import os

# some wrappers for easier migration in the future

_MPI_MASTER_SEND_TAG = 11
_MPI_MASTER_RECEIVE_TAG = 12

if not torch.distributed.is_initialized():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.environ["MULTI_GPU_BACKEND"] = "TORCH_DIST"
    import dally.bert.utils.distributed as dist_utils

    rank = int(os.environ["RANK"])
    # device = rank % torch.cuda.device_count()
    # torch.cuda.set_device(device)


if not torch.distributed.is_initialized():
    print("Something goes totally wrong")


def is_chief():
    return dist_utils.worker_idx() == 0


def worker_idx():
    return dist_utils.worker_idx()


def num_workers():
    return dist_utils.num_workers()


def worker_host_idx():
    return dist_utils.worker_idx() // torch.cuda.device_count()


def worker_local_idx():
    return dist_utils.worker_local_idx()


def is_local_chief():
    return dist_utils.worker_local_idx() == 0


def gpu_visible_device_list():
    return dist_utils.gpu_visible_device_list()


def in_distributed_mode():
    return dist_utils.num_workers() > 1


allreduce = dist_utils.allreduce
allgather = dist_utils.allgather
barrier = dist_utils.barrier
reduce = dist_utils.reduce
