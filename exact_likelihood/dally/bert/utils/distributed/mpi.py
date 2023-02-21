import os
import pickle


try:
    from mpi4py import MPI

    mpicomm = MPI.COMM_WORLD
except ImportError:
    mpicomm = None


class MPIHelper:
    _MPI_MASTER_SEND_TAG = 11
    _MPI_MASTER_RECEIVE_TAG = 12
    _MASTER_WORKER_IDX = 0
    _INT_MAX = 2**31 - 1

    @staticmethod
    def _read_tar_to_buffer(tar_filepath):
        if os.path.exists(tar_filepath):
            data = open(tar_filepath, "rb").read()
            return data
        else:
            return ""

    @staticmethod
    def _send(data, dest, tag):
        def ceil_div(x, y):
            return (x + y - 1) // y

        MAX_MSG_SIZE = (
            MPIHelper._INT_MAX - 1024
        )  # reserve some space for internal mpi4py data
        full_msg_size = len(data)
        n_msgs = ceil_div(full_msg_size, MAX_MSG_SIZE)
        mpicomm.send(n_msgs, dest=dest, tag=tag)
        for i in range(n_msgs):
            begin = i * MAX_MSG_SIZE
            end = min(full_msg_size, (i + 1) * MAX_MSG_SIZE)
            mpicomm.send(data[begin:end], dest=dest, tag=tag)

    @staticmethod
    def _receive(source, tag):
        n_msgs = mpicomm.recv(source=source, tag=tag)
        data = [None for _ in range(n_msgs)]
        for i in range(n_msgs):
            data[i] = mpicomm.recv(source=source, tag=tag)
        return b"".join(data)


def receive_files(base_path, src_rank, tag):
    data2store = MPIHelper._receive(source=src_rank, tag=tag)
    data2store = pickle.loads(data2store)
    write_files(base_path, data2store)
    return list(data2store.keys())


def send_files(data2send, dst_rank, tag, files2transfer):

    data2send = {filepath: data2send[filepath] for filepath in files2transfer}
    data2send = pickle.dumps(data2send)
    MPIHelper._send(data2send, dst_rank, tag)


def write_files(base_path, data2store):
    for filename, element in data2store.items():
        filepath = base_path / filename
        if element.is_binary:
            filepath.write_bytes(element.data)
        else:
            filepath.write_text(element.data)


def sure_relative(filepath, base_path):
    if str(filepath).startswith(str(base_path)):
        return filepath.relative_to(base_path)
    return filepath


def transfer_files(data, base_path, to_master, files2node=None):
    # make all paths relative
    data2send = {}
    for filepath, element in data.items():
        filepath = sure_relative(filepath, base_path)
        data2send[filepath] = element
    if files2node is None:
        all_files = list(data2send.keys())
        files2node = {rank: all_files for rank in range(mpicomm.Get_size())}
    else:
        for dst_rank, filepaths in files2node.items():
            files2node[dst_rank] = [sure_relative(x, base_path) for x in filepaths]
    # some shortcuts
    tag = (
        MPIHelper._MPI_MASTER_RECEIVE_TAG
        if to_master
        else MPIHelper._MPI_MASTER_SEND_TAG
    )
    master_rank = MPIHelper._MASTER_WORKER_IDX
    # transfer data
    received_files = {}
    if mpicomm.Get_rank() == master_rank:
        for rank in range(mpicomm.Get_size()):
            if rank == master_rank:
                continue
            else:
                if to_master:
                    received_files[rank] = receive_files(base_path, rank, tag)
                else:
                    send_files(data2send, rank, tag, files2node[rank])
    else:
        if to_master:
            send_files(data2send, master_rank, tag, files2node[master_rank])
        else:
            received_files[mpicomm.Get_rank()] = receive_files(
                base_path, master_rank, tag
            )
    return received_files


def barrier():
    mpicomm.Barrier()
