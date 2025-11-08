import torch
from .utils import enable_running_stats, disable_running_stats
from torch.distributed import ReduceOp
import contextlib


class SGAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, grad_rho_scheduler=None,
                 grad_norm_rho_scheduler=None, adaptive=False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SGAM, self).__init__(params, defaults)
        self.grad_rho_scheduler = grad_rho_scheduler
        self.grad_norm_rho_scheduler = grad_norm_rho_scheduler
        self.perturb_eps = perturb_eps
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.args = args
        self.get_grad_reduce(grad_reduce)
        self.grad_beta_0 = 0.3 #g_0
        self.grad_beta_1 = 0.3 #g_1
        self.grad_beta_2 = 1 - self.grad_beta_0 #g_2
        self.grad_beta_3 = 1 - self.grad_beta_1 #g_3


        self.update_rho_t()

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        if self.grad_rho_scheduler is not None:
            self.grad_rho = self.grad_rho_scheduler.step()
        if self.grad_norm_rho_scheduler is not None:
            self.grad_norm_rho = self.grad_norm_rho_scheduler.step()

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)  #||\Delta L_{W_t}(\theta_t)||
        scale = self.grad_rho / (grad_norm + self.perturb_eps) # \rho /

        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_0"] = p.grad.data.clone() # set g_0 for p
                    e_w = p.grad * scale.to(p)  # \Delta_L{W_t}(\theta_t)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_0'] = e_w

        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_2"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_1_2'] += e_w

        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone()
                p.grad.data -= self.state[p]["g_0"]


        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.grad_norm_rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                #Line 9 的扰动
                self.state[p]['e_w_0_2'] = e_w
                self.state[p]['e_w_1_2'] = e_w

    @torch.no_grad()
    def unperturb(self, perturb_key: str):

        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])



    @torch.no_grad()
    def gradient_decompose(self, args=None):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]["g_3"] = p.grad.data.clone()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # update the weighted sum of grads
                #h_t,+
                self.state[p]['pro_m'] = self.state[p]['g_0'] + abs(self.grad_beta_2) * self.state[p]['g_2']
                # self.state[p]['pro_m'] = self.grad_beta_1*self.state[p]['g_1'] + self.grad_beta_3 * self.state[p]['g_3']
                # h_t,-
                # self.state[p]['pro_n'] = self.grad_beta_0 * self.state[p]['g_0'] + self.grad_beta_2 * self.state[p][
                #     'g_2']

                p.grad.data = self.grad_beta_1 * self.state[p]["g_1"] + self.grad_beta_3 * p.grad.data.detach().clone()
                inner_prod += torch.sum(
                    self.state[p]['pro_m'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='pro_m')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (
                        new_grad_norm + self.perturb_eps)
                p.grad.data.add_(vertical, alpha=-0.03)


    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False, by: str = 'grad'):
        norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if by == 'grad':
                    g = p.grad.data
                elif by == 'pro_m':
                    g = self.state[p]['pro_m']
                elif by == 'rain_t':
                    g = self.state[p]['rain_t']
                # elif by == 'e_w':
                #     g = self.state[p]['e_w_0'] + self.state[p]['e_w_1_2'] + self.state[p]['e_w_2']
                elif by == 'p':
                    g = p.data
                else:
                    raise ValueError("Invalid 'by' argument in _grad_norm")

                if weight_adaptive:
                    norm += torch.sum((g * torch.abs(p.data)) ** 2)
                else:
                    norm += torch.sum(g ** 2)

        return torch.sqrt(norm)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def update_u_i(self, mu, global_mu):
        ### update g with dynamic dual
        for group in self.param_groups:
            for p, _mu_, _global_mu_ in zip(group["params"], mu, global_mu):
                p.requires_grad = True
                if p.grad is None:
                    continue
                # ▲=g_i-(u_i+s)
                # p.grad.sub_(_mu_ + _global_mu_)
                rain_t=p.grad.data.detach().clone()

                rain_t.sub_(_mu_ + _global_mu_)

                self.state[p]["rain_t"]=rain_t

        # rain_t=torch.cat(rain_t_list)
        #to do
        rain_t_norm=self._grad_norm(by='rain_t')
        ### gradient ascent
        # update s_i
        # g_i范数

        for group in self.param_groups:
            scale =self.grad_rho / (rain_t_norm + 1e-7)
            for p in group["params"]:
                # original SAM
                # e_w = p.grad * scale.to(p)
                # ASAM
                # compute s_i
                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                e_w = 1.0 * self.state[p]["rain_t"] * scale.to(p)
                # climb to the local maximum "w + e(w)"
                # p.add_(e_w * 1)
                # self.state[p]["e_w"]：s_i
                self.state[p]['e_w_2_3'] = e_w + self.state[p]['e_w_0_2']



        ### update local dynamic dual u_i  u_i=u_i+(s_i-s)
        for group in self.param_groups:
            for p, _mu_, _global_mu_ in zip(group["params"], mu, global_mu):
                p.requires_grad = True
                if p.grad is None:
                    continue
                _mu_ += self.state[p]["e_w_2_3"] - _global_mu_

        return mu
    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
            loss.backward()

        self.forward_backward_func = get_grad

    def step(self, u_i,global_s,closure=None):
    # def step(self, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func  #使用数据样本进行前向和后向传播，得到梯度，这里用的是smooth_crossentropy

        with self.maybe_no_sync():
            # get gradient
            # g_t,0
            get_grad() #得到结果、损失值和梯度 h_t^loss

            # perturb weights
            # \
            self.perturb_weights(perturb_idx=0)

            # disable running stats for second pass,
            #
            disable_running_stats(self.model)
            # model 1
            #g_t,1
            get_grad()
            # grad 1

            #
            self.unperturb(perturb_key="e_w_0")
            # model 0
            #h_t,0
            self.grad_norm_ascent()
            # model 2
            #g_t,2
            get_grad()
            # grad 2

            self.perturb_weights(perturb_idx=1)

            #update s_i
            u_i= self.update_u_i(u_i,global_s)
            # model 3
            #g_t,3
            get_grad()
            # grad 3



            # decompose and get new update direction
            self.gradient_decompose()

            # unperturb

            self.unperturb(perturb_key="e_w_1_2")





        # synchronize gradients across workers
        # self._sync_grad()

        # update with new directions
        # self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return u_i

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    # def add_param_group(self, param_group):
    #     self.base_optimizer.add_param_group(param_group)

    def __repr__(self):
        return f'GAM({self.base_optimizer.__class__.__name__})'
