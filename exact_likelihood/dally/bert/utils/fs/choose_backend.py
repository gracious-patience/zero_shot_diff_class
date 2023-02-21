try:
    import nirvana_dl as ndl
except ImportError:
    ndl = None


def running_on_nirvana():
    return ndl is not None


def global_state_fname():
    return "readers_snapshot.pkl"


if running_on_nirvana():
    from bert.utils.fs.nirvana import *
else:
    from bert.utils.fs.default import *
