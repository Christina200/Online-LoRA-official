import torch_optimizer
from torch.nn import Module
from torch import optim
from torch.optim import lr_scheduler

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def select_optimizer(opt_name: str, lr: float, model: Module) -> optim.Optimizer:
    if opt_name == "adam":
        # print("opt_name: adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

def select_scheduler(sched_name: str, opt: optim.Optimizer, hparam=None) -> lr_scheduler._LRScheduler:
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler