
import torch
from torch.optim import Optimizer
import math

class Eve(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999, 0.999),
                 eps=1e-8,
                 k=0.1,
                 K=10.0,
                 weight_decay=0,
                 amsgrad=False):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            k=k,
            K=K,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        if lr <= 0.0:
            raise ValueError
        if not (0.0 <= betas[0] < 1.0) and \
           not (0.0 <= betas[1] < 1.0) and \
           not (0.0 <= betas[2] < 1.0):
            raise ValueError
        if eps <= 0.0:
            raise ValueError
        super(Eve, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Eve, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        loss = None
        f = None
        if closure is not None:
            loss = closure()
            f = loss.data.item()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                lr = group["lr"]
                beta1, beta2, beta3 = group["betas"]
                k = group["k"]
                K = group["K"]
                eps = group["eps"]
                amsgrad = group["amsgrad"]

                state = self.state[p]

                if len(state) == 0:  # init
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format)
                    state["d"] = 1.0
                    if amsgrad:
                        state["max_v"] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format)
                    state["f_hat"] = 0.0

                step = state["step"] + 1
                state["step"] = step

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if group["weight_decay"] != 0:
                    g.add_(group["weight_decay"], p.data)

                m = state["m"]
                m.mul_(beta1).add_(1 - beta1, g)

                v = state["v"]
                v.mul_(beta2).addcmul_(1 - beta2, g, g)

                d = None
                if step > 1:
                    if f < state["f_hat"]:
                        delta = k + 1
                        Delta = K + 1
                    else:
                        delta = 1.0 / (K + 1)
                        Delta = 1.0 / (k + 1)
                    c = min(max(delta, f / state["f_hat"]), Delta)
                    next = c * state["f_hat"]
                    r = np.abs(next - state["f_hat"]) / min(next, state["f_hat"])
                    d = beta3 * state["d"] + (1 - beta3) * r
                    state["d"] = d
                    state["f_hat"] = next
                else:
                    state["f_hat"] = f
                    state["d"] = 1.0
                    d = 1.0

                if amsgrad:
                    max_v = state["max_v"]
                    torch.max(state["max_v"], v, out=max_v)
                    denom = (max_v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / (bias_correction1 * d)

                p.data.addcdiv_(-step_size, m, denom)
        return loss

