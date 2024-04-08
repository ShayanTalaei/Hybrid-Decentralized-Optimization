import copy

from torch.optim import Optimizer
import torch
import torch.autograd.forward_ad as fwAD
from torch.func import functional_call
from torch.func import jvp
from mpi4py import MPI



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

    def __init__(self, params, lr=0.001, random_vec=10, momentum=0.9, names=None, grad_mode='zeroth order simple', v_step=10.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        defaults = dict(lr=lr, random_vec=random_vec, momentum=momentum, names=names, grad_mode=grad_mode, v_step=v_step)
        super(ZAD, self).__init__(params, defaults)
        self.lr = lr
        self.random_vec = random_vec
        self.f = None
        self.momentum = momentum
        self.grad = [torch.zeros(p.size()).to(self.device) for group in self.param_groups for p in group['params']]
        self.params = [p for group in self.param_groups for p in group['params']]
        self.params_data = [p.data for p in self.params]
        self.names = names
        assert grad_mode in ['zeroth_order_simple', 'zeroth_order_forward-mode_AD']
        self.grad_mode = grad_mode
        self.params_dict = {name: p for name, p in zip(self.names, self.params)}
        self.v_step = v_step



    def set_f(self, model, data, target, criterion):
        names = list(n for n, _ in model.named_parameters())

        def f(*params):
            out: torch.Tensor = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, data)
            return criterion(out, target)

        self.f = f

    @torch.no_grad()
    def optimize(self, model, data, target, criterion):
        self.lr = self.param_groups[0]['lr']
        # print('Rank:', MPI.COMM_WORLD.Get_rank(), 'lr:', self.lr)
        # self.set_f(model, data, target, criterion)
        # params = [p for group in self.param_groups for p in group['params']]
        # params_data = [p.data for p in params]
        # total_loss = 0.0
        # torch._foreach_mul_(self.grad, self.momentum)
        # for _ in range(self.random_vec):
        #     v = [torch.rand(p.size()).to(self.device) for p in params_data]
        #     loss, jvp_result = jvp(self.f, tuple(params), tuple(v))
        #     total_loss += loss.item()
        #     torch._foreach_mul_(v, jvp_result.item() * (1 - self.momentum) / self.random_vec)
        #     torch._foreach_add_(self.grad, v)
        #     # torch._foreach_addcmul_(self.grad, v, jvp_result.item() * (1 - self.momentum) / self.random_vec)
        #
        # torch._foreach_add_(params_data, torch._foreach_mul(self.grad, -self.lr))
        # return total_loss / self.random_vec

        if self.grad_mode == 'zeroth_order_forward-mode_AD':
            # print('Rank:', MPI.COMM_WORLD.Get_rank(), 'start')
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
            # print('Rank:', MPI.COMM_WORLD.Get_rank(), 'end')
            return total_loss / self.random_vec

        if self.grad_mode == 'zeroth_order_simple':
            total_loss = 0.0
            torch._foreach_mul_(self.grad, self.momentum)
            for _ in range(self.random_vec):
                v = [torch.randn(p.size()).to(self.device) for p in self.params_data]
                v_norm = torch._foreach_norm(v)
                torch._foreach_add_(v_norm, 1e-8)
                torch._foreach_div_(v, v_norm)
                params_v = copy.deepcopy(self.params_dict)
                for p, v_ in zip(params_v.items(), v):
                    p[1].data += v_ * self.v_step

                lossv = criterion(functional_call(model, params_v, data), target).item()
                loss = criterion(functional_call(model, self.params_dict, data), target).item()

                total_loss += loss
                torch._foreach_mul_(v, (1 - self.momentum) * (lossv - loss) / (self.random_vec * self.v_step))
                torch._foreach_add_(self.grad, v)

            print('Rank:', MPI.COMM_WORLD.Get_rank(), torch.max(torch._foreach_norm(self.grad)))
            torch._foreach_add_(self.params_data, torch._foreach_mul(self.grad, -self.lr))
            return total_loss / self.random_vec

