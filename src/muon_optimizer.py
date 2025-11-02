import torch
from torch.optim import Optimizer

def _approx_orthogonal(W, steps=5):
    if W.ndim != 2:
        raise ValueError("Input must be 2D")

    norm = torch.linalg.norm(W, ord='fro') + 1e-8
    X = W / norm

    for _ in range(steps):
        X = 1.5 * X - 0.5 * X @ X.T @ X
    return X

class Muon(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, steps=5):
        if lr <= 0:
            raise ValueError("Invalid learning rate")
        if not 0 <= momentum <= 1:
            raise ValueError("Invalid momentum")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay")
        if steps < 1:
            raise ValueError("Invalid steps")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, steps=steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            steps = group['steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                grad = grad - p @ (p.T @ grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                M = state['momentum_buffer']
                M.mul_(momentum).add_(grad)

                update = -lr * M
                p.add_(update)
                p.copy_(_approx_orthogonal(p, steps=steps))

        return loss
