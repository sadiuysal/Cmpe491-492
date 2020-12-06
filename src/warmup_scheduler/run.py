# Initializes scheduler to gradually warm-up(increasing) learning rate in optimizer
# Check scheduler.py for source code

import torch

from warmup_scheduler import GradualWarmupScheduler


if __name__ == '__main__':
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = GradualWarmupScheduler(optim, multiplier=8, total_epoch=10)

    for epoch in range(1, 20):
        scheduler.step(epoch)

        print(epoch, optim.param_groups[0]['lr'])
