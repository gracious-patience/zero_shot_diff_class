import logging
import os
from dally.bert.utils.distributed.constants import backend_var_name, horovod_tf_name
from dally.bert.utils.distributed.constants import deepspeed_name, torch_dist_name


if backend_var_name not in os.environ:
    logging.getLogger(__name__).warning(
        "{} environment variable is not set "
        "use {} by default".format(backend_var_name, torch_dist_name)
    )
    os.environ[backend_var_name] = torch_dist_name

if os.environ[backend_var_name] in [deepspeed_name, torch_dist_name]:
    # import deepspeed utils
    from dally.bert.utils.distributed.deepspeed import *
elif os.environ[backend_var_name] == horovod_tf_name:
    raise ValueError("NOT IMPLEMENTED")
else:
    assert False, "Unknown multi-gpu backend type {}".format(
        os.environ[backend_var_name]
    )


def is_chief():
    return worker_idx() == 0
