from copy import deepcopy

from torch.optim import Optimizer
import torch
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call


def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = deepcopy(model_dict[key])
    return new_dict


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

    def __init__(self, params, lr=0.001, random_vec=10, momentum=0.9, names=None, grad_mode='zeroth_order_rge',
                 v_step=10.0, device='cpu', weight_decay=0.0):
        self.device = device
        defaults = dict(lr=lr, random_vec=random_vec, momentum=momentum, names=names, grad_mode=grad_mode,
                        v_step=v_step, weight_decay=weight_decay)
        super(ZAD, self).__init__(list(params), defaults)

        self.lr = lr
        self.random_vec = random_vec
        self.f = None
        self.momentum = momentum
        self.grad = [torch.zeros(p.size()).to(self.device) for group in self.param_groups for p in group['params']]
        self.params = [p for group in self.param_groups for p in group['params']]
        self.weight_decays = [weight_decay if 'weight_decay' not in group else group['weight_decay'] for group in
                              self.param_groups for p in group['params']]

        if len(self.grad) == 0:
            self.grad = [torch.zeros(p.size()).to(self.device) for group in params for p in group['params']]
            self.params = [p for group in params for p in group['params']]
            self.weight_decays = [weight_decay if 'weight_decay' not in group else group['weight_decay'] for group in
                                  params for p in group['params']]
        self.params_data = [p.data for p in self.params]
        self.names = names
        assert grad_mode in ['zeroth_order_rge', 'zeroth_order_forward-mode_AD', 'zeroth_order_cge',
                             'zeroth_order_forward-mode_AD_sim']
        self.grad_mode = grad_mode
        self.params_dict = {name: p for name, p in zip(self.names, self.params)}
        self.v_step = v_step

    def set_f(self, model, data, target, criterion):
        names = list(n for n, _ in model.named_parameters())

        def f(*params):
            out: torch.Tensor = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, data)
            return criterion(out, target)

        self.f = f

    def optimize(self, model, data, target, criterion):
        self.lr = self.param_groups[0]['lr']

        if self.grad_mode == 'zeroth_order_forward-mode_AD':
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                total_loss = 0.0
                for _ in range(self.random_vec):
                    tangents = {name: torch.clip(torch.rand_like(p), min=1e-3) for name, p in self.params_dict.items()}
                    v = [t for t in tangents.values()]

                    dual_params = {}
                    with fwAD.dual_level():
                        for name, p in self.params_dict.items():
                            dual_params[name] = fwAD.make_dual(p, tangents[name])
                        loss = criterion(functional_call(model, dual_params, data), target)
                        jvp_result = fwAD.unpack_dual(loss).tangent
                    torch._foreach_mul_(v, jvp_result.item() * (1 - self.momentum) / self.random_vec)
                    torch._foreach_add_(self.grad, v)
                    total_loss += loss.item()
                torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))
                return total_loss / self.random_vec

        elif self.grad_mode == 'zeroth_order_rge':
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                loss = criterion(functional_call(model, self.params_dict, data), target).item()
                for _ in range(self.random_vec):
                    v = [torch.randn(p.size()).to(self.device) for p in self.params_data]
                    params_v = deepcopy(self.params_dict)
                    for p, v_ in zip(params_v.items(), v):
                        p[1].data += v_ * self.v_step
                    lossv = criterion(functional_call(model, params_v, data), target).item()
                    torch._foreach_mul_(v, (1 - self.momentum) * (lossv - loss) / (self.random_vec * self.v_step))
                    torch._foreach_add_(self.grad, v)
                norms = torch._foreach_norm(self.params_data)
                torch._foreach_mul_(norms, self.weight_decays)
                torch._foreach_mul_(norms, 2)
                torch._foreach_add_(self.grad, norms)

                torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))
                return loss

        elif self.grad_mode == 'zeroth_order_cge':
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                params_v = deepcopy(self.params_dict)
                loss = criterion(functional_call(model, self.params_dict, data), target).item()
                for i, (key, param) in enumerate(self.params_dict.items()):
                    for j in range(param.numel()):
                        if j != 0:
                            params_v[key].data.view(-1)[j - 1] -= self.v_step
                        params_v[key].data.view(-1)[j] += self.v_step
                        loss_v = criterion(functional_call(model, params_v, data), target).item()
                        self.grad[i].view(-1)[j] += (1 - self.momentum) * (loss_v - loss) / self.v_step
                    params_v[key].data.view(-1)[param.numel() - 1] -= self.v_step

                torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))
                return loss

        elif self.grad_mode == 'zeroth_order_forward-mode_AD_sim':
            self.zero_grad()
            loss_target = criterion(model(data), target)
            norms = torch._foreach_norm(self.params_data)
            torch._foreach_pow_(norms, 2)
            torch._foreach_mul_(norms, self.weight_decays)
            loss = torch.sum(torch.tensor(norms)) + loss_target
            loss.backward()
            with torch.no_grad():
                torch._foreach_mul_(self.grad, self.momentum)
                actual_grad = [param.grad if param.grad is not None else torch.zeros(param.size()).to(self.device) for
                               param in self.params]
                for _ in range(self.random_vec):
                    v = [torch.randn(p.size()).to(self.device) for p in self.params_data]
                    efficiency = [t.sum() * (1 - self.momentum) / self.random_vec for t in
                                  torch._foreach_mul(v, actual_grad)]
                    torch._foreach_mul_(v, efficiency)
                    torch._foreach_add_(self.grad, v)
                torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))
                return loss_target.item()
