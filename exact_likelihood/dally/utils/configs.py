import importlib


def instantiate_from_config(config, extra_args=None):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", dict())
    if extra_args:
        return get_obj_from_str(config["target"])(**params, **extra_args)
    else:
        return get_obj_from_str(config["target"])(**params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
