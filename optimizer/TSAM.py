import torch
import torch.nn.functional as F


class TSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10
        self.kai = 1

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(TSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.param_groups[0]["rho"] = rho
        self.param_groups[0]["adaptive"] = adaptive
        for p1 in self.param_groups[0]["params"]:
            self.state[p1]["last_g"] = 0
        self.paras = None

    @torch.no_grad()
    def first_step(self):
        # first order sum
        grad_norm = self._grad_norm()  # the norm of g_{i,k}^t + g_{i,k-1}^t + self.kai * delta
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-7)
        for p1,p2  in zip(self.param_groups[0]["params"],self.param_groups[1]["params"]):
            p1.requires_grad = True
            if p1.grad is None:
                continue
            # original SAM
            # e_w = p.grad * scale.to(p)
            # ASAM
            e_w = (torch.pow(p1, 2) if self.param_groups[0]["adaptive"] else 1.0) * (p1.grad + self.kai * p2 + self.state[p1]["last_g"]) * scale.to(p1)
            # climb to the local maximum "w + e(w)"
            p1.add_(e_w * 1)
            self.state[p1]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, alpha=1.):
        # model.require_backward_grad_sync = False
        # model.require_forward_param_sync = True
        inputs, labels, loss_func, model = self.paras

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.first_step()
        # model.require_backward_grad_sync = True
        # model.require_forward_param_sync = False

        predictions = model(inputs)
        loss = alpha * loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.get_last_gradient()

        self.second_step()

    def _grad_norm(self):
        norm = torch.norm(torch.stack([
            # original SAM
            # p.grad.norm(p=2).to(shared_device)
            # ASAM
            (p1.grad + self.state[p1]["last_g"] + self.kai * p2).norm(p=2)
            for p1,p2 in zip(self.param_groups[0]["params"], self.param_groups[1]["params"]) if p1 is not None and p2 is not None]), p=2)
        return norm

    def get_last_gradient(self):
        for p1 in self.param_groups[0]["params"]:
            if p1.grad is None:
                continue
            self.state[p1]["last_g"] = p1.grad