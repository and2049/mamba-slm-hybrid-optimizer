import torch
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer

def _approx_orthogonal(M, steps=5):
    if M.ndim != 2:
        raise ValueError("Newton-Schulz requires 2D matrix")

    norm = torch.linalg.norm(M, ord='fro') + 1.e-8
    X = M / norm

    for _ in range(steps):
        X = 1.5 * X - 0.5 * X @ X.T @ X

    O = X
    return O

class Muon(Optimizer):

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0, steps = 5, nesterov=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if steps < 1:
            raise ValueError("Invalid steps: {}".format(steps))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, steps=steps, nesterov=nesterov)
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            steps = group['steps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')

                if p.ndim != 2:
                    print(f"Muon optimizer recieved non-2D parameter in shape {p.ndim}. Muon will be skipped for this parameter")
                    continue

                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                M = state['momentum_buffer']

                # update logic

                if weight_decay != 0:
                    grad.add_(p, alpha=weight_decay)

                M.mul_(momentum).add(grad)

                if nesterov:
                    O = _approx_orthogonal(M, steps=steps)
                else:
                    O = _approx_orthogonal(M, steps=steps)

                p.add_(O, alpha=-lr)

        return loss