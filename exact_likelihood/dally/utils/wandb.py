import wandb
import dally.utils.ddp as ddp


class WandbLogger:
    def __init__(self, entity, project):
        wandb.init(entity=entity, project=project, group="DDP")

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)
