try:
    import nirvana_dl as ndl
except:
    # This should not happend
    # You must not import this module directly
    ndl = None


def data_dir():
    return ndl.input_data_path()


def checkpoint_dir():
    return ndl.snapshot.get_snapshot_path()


def logs_dir():
    return ndl.logs_path()


def output_dir():
    return ndl.output_data_path()


def json_output_file():
    return ndl.json_output_file()


def params():
    try:
        return ndl.params()
    except RuntimeError:
        import logging

        logging.getLogger().info(
            "Running on nirvana, but couldn't get params. "
            'Either both or none of "params" input and option was set.'
        )
