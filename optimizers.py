import copy

from torch.optim import Optimizer
import torch
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call
from torch.func import jvp



class SGD(torch.optim.SGD):
    name = 'SGD'

    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        return super(SGD, self).step(closure)

    def optimize(self, model, data, target, criterion):
        self.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        self.step()
        return loss.item()


class ZAD(Optimizer):
    name = 'ZAD'

    def __init__(self, params, lr=0.001, random_vec=100, momentum=0.9):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        defaults = dict(lr=lr, random_vec=random_vec, momentum=momentum)
        super(ZAD, self).__init__(params, defaults)
        self.lr = lr
        self.random_vec = random_vec
        self.f = None
        self.momentum = momentum
        self.grad = [torch.zeros(p.size()).to(self.device) for group in self.param_groups for p in group['params']]

    def set_f(self, model, data, target, criterion):
        names = list(n for n, _ in model.named_parameters())

        def f(*params):
            out: torch.Tensor = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, data)
            return criterion(out, target)

        self.f = f

    @torch.no_grad()
    def optimize(self, model, data, target, criterion):
        self.set_f(model, data, target, criterion)
        params = [p for group in self.param_groups for p in group['params']]
        params_data = [p.data for p in params]
        total_loss = 0.0
        torch._foreach_mul_(self.grad, self.momentum)
        for _ in range(self.random_vec):
            v = [torch.rand(p.size()).to(self.device) for p in params_data]
            loss, jvp_result = jvp(self.f, tuple(params), tuple(v))
            total_loss += loss.item()
            # torch._foreach_mul_(v, jvp_result.item() * (1 - self.momentum) / self.random_vec)
            torch._foreach_addcmul_(self.grad, v, jvp_result.item() * (1 - self.momentum) / self.random_vec)

        torch._foreach_addcmul_(params_data, self.grad, -self.lr)
        return total_loss / self.random_vec
        # params = {name: p for name, p in model.named_parameters()}
        # tangents = {name: torch.clip(torch.rand_like(p), min=1e-5) for name, p in params.items()}
        #
        # dual_params = {}
        # with fwAD.dual_level():
        #     for name, p in params.items():
        #         # Using the same ``tangents`` from the above section
        #         dual_params[name] = fwAD.make_dual(p, tangents[name])
        #     out = functional_call(model, dual_params, data)
        #     jvp = fwAD.unpack_dual(out).tangent
        #
        #     loss = criterion(out, target)
        #     return loss.item()
