import functools
import torch.optim as optim
from torch.optim import lr_scheduler


def optim_builder(optimizer_type, optimizer_kwargs):
    optimizer_class = getattr(optim, optimizer_type)
    return functools.partial(optimizer_class, **optimizer_kwargs)


def schedule_builder(schedule_type, schedule_kwargs):
    schedule_class = getattr(lr_scheduler, schedule_type)
    return functools.partial(schedule_class, **schedule_kwargs)
