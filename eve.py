
import torch
from torch.optim import Optimizer
import math

class Eve(Optimizer):
    '''
    Implementation of Eve(https://arxiv.org/pdf/1611.01505.pdf)

    example:
    ```python
    model = HogeModel()
    criterion = nn.L1loss()
    optimizer = Eve(model.parameters())

    optimizer.zero_grad()
    input = torch.rand((2, 3))
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step(lambda: loss)  # Note: lambda expression is required
    ```
    '''
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999, 0.999),
                 c=10.0,
                 eps=1e-8,
                 f_star=0.0,
                 weight_decay=0,
                 amsgrad=False):
        defaults = dict(
            lr=lr,
            betas=betas,
            c=c,
            eps=eps,
            f_star=f_star,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        if lr <= 0.0:
            raise ValueError
        if not (0.0 <= betas[0] < 1.0) and \
           not (0.0 <= betas[1] < 1.0) and \
           not (0.0 <= betas[2] < 1.0):
            raise ValueError
        if c < 1.0:
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
            f = loss.data
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                lr = group["lr"]
                beta1, beta2, beta3 = group["betas"]
                c = group["c"]
                eps = group["eps"]
                f_star = group["f_star"]
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
                    if amsgrad:
                        state["max_v"] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format)

                step = state["step"] + 1
                state["step"] = step

                bias_correction1 = 1 - math.power(beta1, step)
                bias_correction2 = 1 - math.power(beta2, step)

                if group["weight_decay"] != 0:
                    g.add_(group["weight_decay"], p.data)

                m = state["m"]
                m.mul_(beta1).add_(1 - beta1, g)

                v = state["v"]
                v.mul_(beta2).addcmul_(1 - beta2, g, g)

                d_tilde = None
                if step > 1:
                    prev_f = state["f"]
                    state["f"] = f
                    d = torch.abs(f - prev_f) / (torch.min(f, prev_f) - f_star)
                    d_hat = d.clamp(1 / c, c)
                    d_tilde = beta3 * state["d_tilde"] + (1 - beta3) * d_hat
                    state["d_tilde"] = d_tilde
                else:
                    state["f"] = f
                    d_tilde = 1.0
                    state["d_tilde"] = d_tilde

                v = v if not amsgrad else torch.max(
                        state["max_v"], v, out=state["max_v"])
                p.data.addcdiv_(
                    -lr if d_tilde == 1.0 else -lr / d_tilde,
                    m * math.sqrt(bias_correction2),
                    v.sqrt()
                     .add(eps, math.sqrt(bias_correction2))
                     .mul(bias_correction1)
                    )

        return loss

