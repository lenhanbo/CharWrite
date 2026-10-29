import torch

from charwritevit.train.utils import make_scaler, make_warmup_cosine_scheduler, optimizer_step


def test_warmup_cosine_steps_only_with_optimizer_updates():
    parameter=torch.nn.Parameter(torch.tensor(1.0))
    optimizer=torch.optim.SGD([parameter],lr=1.0)
    scaler=make_scaler(torch.device("cpu"),False)
    scheduler=make_warmup_cosine_scheduler(optimizer,total_steps=6,warmup_steps=2,min_lr_ratio=0.1)
    initial_lr=optimizer.param_groups[0]["lr"]
    assert 0 < initial_lr < 1.0
    assert not optimizer_step(parameter.square(),optimizer,scaler,2,0)
    assert optimizer.param_groups[0]["lr"]==initial_lr
    assert optimizer_step(parameter.square(),optimizer,scaler,2,1)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"]==1.0
    for step in range(2,12):
        if optimizer_step(parameter.square(),optimizer,scaler,2,step): scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"]-0.1)<1e-6
