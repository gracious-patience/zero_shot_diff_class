import os
from os.path import dirname as up


_BASE_PATH_DIRECTORY = up(up(up(os.path.realpath(__file__))))


def set_base_dir(path):
    global _BASE_PATH_DIRECTORY
    _BASE_PATH_DIRECTORY = str(path)


def _sure_dir_exists(path):
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def sure_all_dirs_exist():
    _sure_dir_exists(checkpoint_dir())
    _sure_dir_exists(logs_dir())
    _sure_dir_exists(output_dir())


def data_dir():
    return os.path.join(_BASE_PATH_DIRECTORY, "data")


def checkpoint_dir():
    return os.path.join(_BASE_PATH_DIRECTORY, "checkpoint")


def logs_dir():
    return os.path.join(_BASE_PATH_DIRECTORY, "checkpoint/logs")


def output_dir():
    return os.path.join(_BASE_PATH_DIRECTORY, "output")


def params():
    return {}
