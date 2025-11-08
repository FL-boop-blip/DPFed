import math

import numpy as np
import torch
from opacus import PrivacyEngine

from optimizer.Adan import Adan
from optimizer.GAM import GAM
from optimizer.GAM_KL import GAM_KL
from optimizer.SAM import SAM
from optimizer.utils import ProportionScheduler
from utils_libs import *
from utils_dataset import *
from utils_models import *
from optimizer.ESAM import ESAM
from optimizer.TSAM import TSAM
from optimizer.DRegSAM import DRegSAM
from scipy import special

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

import time

max_norm = 10


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T ** 2)
        return loss

def compute_client_gradient(global_model,trn_x,trn_y,batch_size,dataset_name):
    total_gradient = None
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    trn_gen_iter = trn_gen.__iter__()
    for i in range(int(np.ceil(n_trn / batch_size))):
        batch_x, batch_y = trn_gen_iter.__next__()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        y_pred = global_model(batch_x)
        loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
        loss = loss_f_i
        global_model.zero_grad()
        loss.backward()
        current_grad = np.concatenate([
            param.grad.detach().cpu().numpy().flatten() for param in global_model.parameters()
        ])

        if total_gradient is None:
            total_gradient = current_grad.copy()
        else:
            total_gradient += current_grad

    return total_gradient

def compute_D(clnt_gradients,weight_list,n_clnt):
    g_gradient = np.zeros_like(clnt_gradients[0])
    for i in range(len(clnt_gradients)):
        g_gradient += 1/n_clnt * clnt_gradients[i]
    total_gradient_norm = np.linalg.norm(g_gradient, 2)**2
    print('total_gradient_norm is {}'.format(total_gradient_norm))
    normalized_sums = 0.0
    for i, grad in enumerate(clnt_gradients):
        grad_norm_squared = np.linalg.norm(grad, 2)**2
        print('client {}, gradient_norm is {}'.format(i,grad_norm_squared))
        print("weight list {} is {}".format(i,weight_list[i]))
        normalized_sums += 1/n_clnt * (grad_norm_squared / total_gradient_norm)
    return np.sqrt(normalized_sums)



def sinp(epoch,t):
    return 0.5 * (1+ math.sin(math.pi * t / epoch))



def get_update_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)


def get_params_list_with_shape(model, param_list, device):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape).to(device))
    return vec_with_shape


def get_mdl_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)


def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)


def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(participation_clients), :]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:, np.newaxis]

    if metric == 'cosine':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric == 'only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores > 0.9, 0.01, float('inf'))
    elif metric == 'l1':
        difference = np.linalg.norm(local_distributions - hypo_distribution, ord=1, axis=1)
    elif metric == 'l2':
        difference = np.linalg.norm(local_distributions - hypo_distribution, axis=1)
    elif metric == 'kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(
            difference)
    return difference


def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_tmp = old_weight - a * distribution_difference + b

    if np.sum(weight_tmp > 0) > 0:
        new_weight = np.copy(weight_tmp)
        new_weight[new_weight < 0.0] = 0.0
    else:
        new_weight = np.copy(old_weight)

    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst

def train_model_DPFER(model, model_func, cur_cld_model, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                       learning_rate, batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                              batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_div = DistillKL(4.0)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(torch.optim.SGD(model.parameters(), lr=learning_rate),
                                                step_size=sch_step, gamma=sch_gamma)

    # === AdDPNFL-specific parameters (internal defaults) === #
    sigma = 0.1  # noise scale
    beta1 = 0.9  # momentum 1
    beta2 = 0.99  # momentum 2
    pi = 1e-3  # small constant
    alpha_l = learning_rate  # reuse existing LR as local LR

    def get_flat_params(model):
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def set_flat_params(model, flat_params):
        pointer = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
            pointer += numel

    w_global = get_flat_params(model).clone().detach()
    mu_t = torch.zeros_like(w_global).to(device)
    nu_t = torch.ones_like(w_global).to(device) * (pi ** 2)

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        epoch_loss = 0
        trn_gen_iter = iter(trn_gen)
        for _ in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = next(trn_gen_iter)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            y_pred = model(batch_x)
            y_t = cur_cld_model(batch_x)

            # Supervised + distillation loss
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long()) / batch_y.size(0)
            loss_kl = loss_div(y_pred, y_t)
            loss = loss_f_i + 1.0 * loss_kl

            # Compute gradient
            model.zero_grad()
            loss.backward()
            grads = torch.cat([p.grad.view(-1) for p in model.parameters()])

            # === DP核心：clip + noise ===
            grad_norm = torch.norm(grads, p=2)
            clipped_grad = grads / max(1.0, grad_norm / max_norm)
            noise = torch.randn_like(clipped_grad) * sigma
            noisy_grad = clipped_grad + noise

            # Momentum-style DP update
            mu_t = beta1 * mu_t + (1 - beta1) * noisy_grad
            nu_t = beta2 * nu_t + (1 - beta2) * (noisy_grad ** 2)

            alpha_l = scheduler.get_last_lr()[0]
            delta = -alpha_l * mu_t / (torch.sqrt(nu_t) + pi)
            new_params = w_global + delta
            set_flat_params(model, new_params)

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay is not None:
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f" %
                  (e + 1, epoch_loss, scheduler.get_lr()[0]))
            model.train()
        scheduler.step()

    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model





def train_model_MoFedSAM(model, model_func, alpha_coef, delta_list, trn_x, trn_y,
                       learning_rate, rho,batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=rho)
    model.train();
    model = model.to(device)
    w_global = get_flat_params(model).clone().detach()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step(alpha=1)


            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = (1-alpha_coef) * torch.sum(local_par_list * delta_list)


            ###
            loss_algo.backward()

            ###
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss_algo.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
    dp_sigma = 0.95
    dp_clip = 0.2
    clnt = 100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta * min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita = dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
def train_model_FedAdan(model, model_func, alpha_coef, delta_list, trn_x, trn_y,
                       learning_rate, rho,batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # optimizer = Adan(model.parameters())
    optimizer = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]


            # Get linear penalty on the current parameter estimates

            loss = loss_f_i

            ###
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
# --- Helper functions
def train_model_FedSGAM_KL(cur_cld_model,model, model_func, alpha_coef,
                        avg_mdl_param,
                        hist_params_diff,
                        trn_x, trn_y,
                        learning_rate, batch_size, epoch, print_per,
                        weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
    kd_T=4
    loss_div = DistillKL(kd_T)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    grad_norm_rho = 0.2
    grad_rho = 0.03
    grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.1, min_lr=0.0,
                                             max_value=grad_rho, min_value=grad_rho)

    grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.1, min_lr=0.0,
                                                  max_value=grad_norm_rho, min_value=grad_norm_rho)

    rho_optimizer = GAM_KL(params=model.parameters(), base_optimizer=optimizer, model=model,
                        grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                        adaptive=False)
    w_global = get_flat_params(model).clone().detach()
    model.train()
    model = model.to(device)

    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()


            y_t = cur_cld_model(batch_x)

            rho_optimizer.set_closure(loss_fn, batch_x, batch_y,loss_div,y_t)

            rho_optimizer.step()

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding
            rho_optimizer.update_rho_t()
            # optimizer.step()
            # optimizer.zero_grad()

            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            # loss function
            loss_correct = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss_correct.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            optimizer.zero_grad()

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
    dp_sigma = 0.95
    dp_clip = 0.2
    clnt = 100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta * min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita = dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
def train_model_FedGAM_KL(cur_cld_model,model, model_func, alpha_coef,
                        avg_mdl_param,
                        hist_params_diff,
                        trn_x, trn_y,
                        learning_rate, batch_size, epoch, print_per,
                        weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
    kd_T=4
    loss_div = DistillKL(kd_T)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    grad_norm_rho = 0.2
    grad_rho = 0.03
    grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.1, min_lr=0.0,
                                             max_value=grad_rho, min_value=grad_rho)

    grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.1, min_lr=0.0,
                                                  max_value=grad_norm_rho, min_value=grad_norm_rho)

    rho_optimizer = GAM(params=model.parameters(), base_optimizer=optimizer, model=model,
                        grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                        adaptive=False)
    w_global = get_flat_params(model).clone().detach()
    model.train()
    model = model.to(device)

    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()


            y_t = cur_cld_model(batch_x)

            rho_optimizer.set_closure(loss_fn, batch_x, batch_y,loss_div,y_t)

            rho_optimizer.step()

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding
            rho_optimizer.update_rho_t()
            # optimizer.step()
            # optimizer.zero_grad()

            # local_par_list = None
            # for param in model.parameters():
            #     if not isinstance(local_par_list, torch.Tensor):
            #         # Initially nothing to concatenate
            #         local_par_list = param.reshape(-1)
            #     else:
            #         local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            # # loss function
            # loss_correct = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            #
            # loss_correct.backward()
            #
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding
            # optimizer.step()
            # optimizer.zero_grad()

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
    dp_sigma = 0.95
    dp_clip = 0.2
    clnt = 10
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta * min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita = dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node));
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].named_parameters()))

    param_0 = clnt_models[0].named_parameters()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl


def train_model_FedSGAM(model, model_func, alpha_coef,
                        u_i, global_s, avg_mdl_param,
                        hist_params_diff,
                        trn_x, trn_y,
                        learning_rate, batch_size, epoch, print_per,
                        weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    grad_norm_rho = 0.2
    grad_rho = 0.03
    grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.1, min_lr=0.0,
                                             max_value=grad_rho, min_value=grad_rho)

    grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=0.1, min_lr=0.0,
                                                  max_value=grad_norm_rho, min_value=grad_norm_rho)

    rho_optimizer = GAM(params=model.parameters(), base_optimizer=optimizer, model=model,
                        grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                        adaptive=False)

    model.train()
    model = model.to(device)

    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.set_closure(loss_fn, batch_x, batch_y)

            u_i = rho_optimizer.step(u_i, global_s)

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding
            rho_optimizer.update_rho_t()
            # optimizer.step()
            # optimizer.zero_grad()

            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            # loss function
            loss_correct = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss_correct.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            optimizer.zero_grad()

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    # loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    # if print_test:
    #     loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
    #     print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
    #           %(0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    # else:
    #     print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
    #           %(0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size - 1))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def train_model_FedDDL(p,model, cur_cld_model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                    learning_rate, batch_size, epoch, kd_T,print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_div = DistillKL(kd_T)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
    rho=0.03
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay+alpha_coef)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=rho)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # y_pred = model(batch_xfedsgam
            # feat_s, y_pred = model(batch_x, is_feat=True)
            # feat_t, y_t = cur_cld_model(batch_x, is_feat=True)
            y_pred = model(batch_x)
            y_t = cur_cld_model(batch_x)

            rho_optimizer.paras = [batch_x,batch_y,loss_fn,model]
            rho_optimizer.step(loss_div,y_t)

            ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            # loss_f_i = loss_f_i / list(batch_y.size())[0]
            loss_k_i = loss_div(y_pred, y_t)

            # Get linear penalty on the current parameter estimates
            # local_par_list = None
            # for param in model.parameters():
            #     if not isinstance(local_par_list, torch.Tensor):
            #         # Initially nothing to concatenate
            #         local_par_list = param.reshape(-1)
            #     else:
            #         local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            #
            # loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            #
            # # loss = loss_f_i + loss_algo + p * loss_k_i
            # loss = loss_algo
            # ###
            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding
            # optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            # loss function
            loss_correct = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss_correct.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            optimizer.zero_grad()

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model





def train_model_avg(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                    dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
        pointer += numel



def train_model_AdDPNFL(model, avg_model,trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                    dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # dp_delta = 1e-5
    # ➤ 插入 Opacus 的差分隐私训练机制
    # privacy_engine = PrivacyEngine()
    # model, optimizer, trn_gen = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=trn_gen,
    #     noise_multiplier=0.8,
    #     max_grad_norm=0.8,
    # )
    n_par = get_mdl_params([model]).shape[1]
    w_global = get_mdl_params([model], n_par)[0]
    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
    # === DP核心：clip + noise ===
    dp_sigma=0.95
    dp_clip=0.2
    clnt=100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_mdl_params([model], n_par)[0] # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = torch.tensor(w_local - w_global)
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta* min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita= dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = torch.tensor(w_global) + torch.tensor(noisy_delta)  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze and return
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    noisy_delta=noisy_delta.numpy()

    return model, noisy_delta

def train_model_AdDPNFL_KL(model, avg_model,trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                    dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_div = DistillKL(4.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # dp_delta = 1e-5
    # ➤ 插入 Opacus 的差分隐私训练机制
    # privacy_engine = PrivacyEngine()
    # model, optimizer, trn_gen = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=trn_gen,
    #     noise_multiplier=0.8,
    #     max_grad_norm=0.8,
    # )
    n_par = get_mdl_params([model]).shape[1]
    w_global = get_mdl_params([model], n_par)[0]
    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            y_pred = model(batch_x)
            y_t = avg_model(batch_x)

            # Supervised + distillation loss
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long()) / batch_y.size(0)
            loss_kl = loss_div(y_pred, y_t)
            loss = loss_f_i + 0.5 * loss_kl

            # loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
    # === DP核心：clip + noise ===
    dp_sigma=0.95
    dp_clip=0.2
    clnt=100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_mdl_params([model], n_par)[0] # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = torch.tensor(w_local - w_global)
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta* min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita= dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = torch.tensor(w_global) + torch.tensor(noisy_delta)  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze and return
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    noisy_delta=noisy_delta.numpy()

    return model, noisy_delta

def train_model_DPavg(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                    dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # dp_delta = 1e-5
    # ➤ 插入 Opacus 的差分隐私训练机制
    # privacy_engine = PrivacyEngine()
    # model, optimizer, trn_gen = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=trn_gen,
    #     noise_multiplier=0.8,
    #     max_grad_norm=0.8,
    # )
    w_global = get_flat_params(model).clone().detach()
    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
    # epsilon = privacy_engine.accountant.get_epsilon(delta=dp_delta)
    # print("epsilon:")
    # print(epsilon)
    # Freeze model


    # === DP核心：clip + noise ===
    dp_sigma=0.95
    dp_clip=0.2
    clnt=100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta* min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita= dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)

    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_prox(model, cld_mdl_param, alpha_coef, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch,
                     print_per, weight_decay, dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):

            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef / 2 * torch.sum((local_par_list - cld_mdl_param) * (local_par_list - cld_mdl_param))

            loss = loss / list(batch_y.size())[0] + loss_algo
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y,
                       learning_rate, batch_size, n_minibatch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = Adan(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_trn / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    count_step = 0
    is_done = False

    step_loss = 0;
    n_data_step = 0
    for e in range(epoch):
        # Training
        ###
        if is_done:
            break
        ###        

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            ####
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            ###
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * state_params_diff)

            loss = loss_f_i + loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding


            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay) / 2 * np.sum(params * params)

                print("Step %3d, Training Loss: %.4f, LR: %.5f"
                      % (count_step, step_loss, scheduler.get_lr()[0]))
                step_loss = 0;
                n_data_step = 0

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def train_DPscaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y,
                       learning_rate, batch_size, n_minibatch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # dp_delta = 1e-5
    # # ➤ 插入 Opacus 的差分隐私训练机制
    # privacy_engine = PrivacyEngine()
    # model, optimizer, trn_gen = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=trn_gen,
    #     noise_multiplier=0.8,
    #     max_grad_norm=0.8,
    # )
    w_global = get_flat_params(model).clone().detach()
    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_trn / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    count_step = 0
    is_done = False

    step_loss = 0;
    n_data_step = 0
    for e in range(epoch):
        # Training
        ###
        if is_done:
            break
        ###

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            ####
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            ###
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * state_params_diff)

            loss = loss_f_i + loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding


            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay) / 2 * np.sum(params * params)

                print("Step %3d, Training Loss: %.4f, LR: %.5f"
                      % (count_step, step_loss, scheduler.get_lr()[0]))
                step_loss = 0;
                n_data_step = 0

            model.train()
        scheduler.step()
    # 获取 DP 隐私预算 ε
    # epsilon = privacy_engine.accountant.get_epsilon(delta=dp_delta)
    # print("epsilon:")
    # print(epsilon)
    # best_alpha = None  # PRVAccountant 没有 alpha 参数

    # === DP核心：clip + noise ===
    dp_sigma=0.95
    dp_clip=0.2
    clnt=10
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta* min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita= dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
    # return model


def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def train_model_FedDyn(model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                       learning_rate, batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss = loss_f_i + loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_FedDC(model, model_func, alpha, local_update_last, global_update_last, global_model_param, hist_i,
                      trn_x, trn_y,
                      learning_rate, batch_size, epoch, print_per,
                      weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff)

            loss = loss_f_i + loss_cp + loss_cg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_Fedspeed(model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                         learning_rate, rho, batch_size, epoch, print_per,
                         weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=rho)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_correct = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss_correct.backward()

            ###
            # optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_FedAvr(model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                       learning_rate, batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma, lamda):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            # loss_algo = alpha_coef * torch.sum(hist_params_diff*hist_params_diff)
            # loss_quad = torch.sum(lamda * hist_params_diff)
            loss_quad = F.mse_loss(lamda, hist_params_diff, reduction="mean")

            loss = loss_f_i + loss_algo - alpha_coef * loss_quad
            # loss = loss_f_i + loss_algo - loss_quad
            # print(loss)
            # print(loss_f_i)
            # print(loss_algo)
            # print(loss_quad)

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
def train_model_DPFedsmoo(model, model_func, alpha_coef, Dynamic_dual, Dynamic_dual_correction, avg_mdl_param,
                        hist_params_diff, trn_x, trn_y,
                        learning_rate, rho, batch_size, epoch, print_per,
                        weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
    rho_optimizer = DRegSAM(model.parameters(), optimizer, rho=rho)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    w_global = get_flat_params(model).clone().detach()
    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model, Dynamic_dual_correction]
            # Dynamic_dual:u_i
            dynamic_dual = rho_optimizer.step(Dynamic_dual)

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            # loss function
            loss_correct = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss_correct.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # === DP核心：clip + noise ===
    dp_sigma = 0.95
    dp_clip = 0.2
    clnt = 100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta * min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita = dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model, dynamic_dual


def train_model_Fedsmoo(model, model_func, alpha_coef, Dynamic_dual, Dynamic_dual_correction, avg_mdl_param,
                        hist_params_diff, trn_x, trn_y,
                        learning_rate, rho, batch_size, epoch, print_per,
                        weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
    rho_optimizer = DRegSAM(model.parameters(), optimizer, rho=rho)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model, Dynamic_dual_correction]
            # Dynamic_dual:u_i
            dynamic_dual = rho_optimizer.step(Dynamic_dual)

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            # loss function
            loss_correct = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss_correct.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model, dynamic_dual


def train_model_Fedtoga(model, model_func, alpha_coef, beta, avg_mdl_param, hist_params_diff, delta, trn_x, trn_y,
                        learning_rate, batch_size, epoch, print_per,
                        weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    delta_model_sel = set_client_from_params(model_func().to(device), delta)

    for param in delta_model_sel.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
    optimizer.add_param_group({'params': delta_model_sel.parameters()})
    rho_optimizer = TSAM(model.parameters(), optimizer, rho=0.1)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            rho_optimizer.paras = [batch_x, batch_y.reshape(-1).long(), loss_fn, model]
            rho_optimizer.step()

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            #
            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            delta_par_list = None
            for param1, param2 in zip(model.parameters(), delta_model_sel.parameters()):
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param1.reshape(-1)
                    delta_par_list = param2.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param1.reshape(-1)), 0)
                    delta_par_list = torch.cat((delta_par_list, param2.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            loss_delta = beta * torch.sum(local_par_list * delta_par_list)

            loss_correct = loss_algo + loss_delta

            loss_correct.backward()

            ###
            # optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_pre(model, model_func, trn_x, trn_y, global_mdl, learning_rate, batch_size, epoch, print_per,
                    weight_decay,
                    dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    n_par = len(get_mdl_params([model_func()])[0])

    model.train()

    for e in range(epoch):

        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss_f_i
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_FedASK(model, model_func, alpha, global_model_param, parameter_drifts_curr,
                       trn_x, trn_y,
                       learning_rate, batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=0.1)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            #
            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - parameter_drifts_curr)) * (
                        local_parameter - (global_model_param - parameter_drifts_curr)))

            loss_correct = loss_cp
            loss_correct.backward()

            # loss = loss_f_i + loss_cp + loss_cg
            # optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


# def train_model_FedLESAM_D(model, model_func, alpha_coef, cld_mdl_param, hist_params_diff, g_update, trn_x, trn_y,
#                            learning_rate, rho, batch_size, epoch, print_per,
#                            weight_decay, dataset_name, sch_step, sch_gamma):
#     n_trn = trn_x.shape[0]
#
#     trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
#                               shuffle=True)
#
#     # loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
#
#     def loss_dyn(predictions, labels, param_list, delta_list, lamb):
#         return torch.nn.functional.cross_entropy(predictions, labels, reduction='mean') + torch.sum(
#             param_list * delta_list) * lamb
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
#     rho_optimizer = LESAM_D(model.parameters(), optimizer, rho=rho)
#
#     model.train()
#     model = model.to(device)
#
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
#     model.train()
#
#     n_par = get_mdl_params([model_func()]).shape[1]
#     delta_list = hist_params_diff - cld_mdl_param
#
#     for e in range(epoch):
#         # Training
#         epoch_loss = 0
#         trn_gen_iter = trn_gen.__iter__()
#         for i in range(int(np.ceil(n_trn / batch_size))):
#             batch_x, batch_y = trn_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device).reshape(-1).long()
#             rho_optimizer.paras = [batch_x, batch_y, loss_dyn, model, delta_list, alpha_coef]
#             rho_optimizer.step(g_update)
#
#             # Get linear penalty on the current parameter estimates
#
#             ###
#             # optimizer.zero_grad()
#             # loss.backward()
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
#                                            max_norm=max_norm)  # Clip gradients to prevent exploding
#             optimizer.step()
#             # epoch_loss += loss.item() * list(batch_y.size())[0]
#
#         if (e + 1) % print_per == 0:
#             epoch_loss /= n_trn
#             if weight_decay != None:
#                 # Add L2 loss to complete f_i
#                 params = get_mdl_params([model], n_par)
#                 epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
#
#             print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
#                   % (e + 1, epoch_loss, scheduler.get_lr()[0]))
#
#             model.train()
#         scheduler.step()
#
#     # Freeze model
#     for params in model.parameters():
#         params.requires_grad = False
#     model.eval()
#
#     return model


# def train_model_FedLESAM(model, model_func, alpha_coef, cld_update_param, trn_x, trn_y,
#                          learning_rate, rho, batch_size, epoch, print_per,
#                          weight_decay, dataset_name, sch_step, sch_gamma):
#     n_trn = trn_x.shape[0]
#
#     trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
#                               shuffle=True)
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
#     rho_optimizer = LESAM_D(model.parameters(), optimizer, rho=rho)
#
#     model.train()
#     model = model.to(device)
#
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
#     model.train()
#
#     n_par = get_mdl_params([model_func()]).shape[1]
#
#     for e in range(epoch):
#         # Training
#         epoch_loss = 0
#         trn_gen_iter = trn_gen.__iter__()
#         for i in range(int(np.ceil(n_trn / batch_size))):
#             batch_x, batch_y = trn_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device).reshape(-1).long()
#             rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
#             rho_optimizer.step(cld_update_param)
#
#             ###
#             # optimizer.zero_grad()
#             # loss.backward()
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
#                                            max_norm=max_norm)  # Clip gradients to prevent exploding
#             optimizer.step()
#             # epoch_loss += loss.item() * list(batch_y.size())[0]
#
#         if (e + 1) % print_per == 0:
#             epoch_loss /= n_trn
#             if weight_decay != None:
#                 # Add L2 loss to complete f_i
#                 params = get_mdl_params([model], n_par)
#                 epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
#
#             print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
#                   % (e + 1, epoch_loss, scheduler.get_lr()[0]))
#
#             model.train()
#         scheduler.step()
#
#     # Freeze model
#     for params in model.parameters():
#         params.requires_grad = False
#     model.eval()
#
#     return model


# def train_model_FedLESAM_S(model, model_func, alpha_coef, state_params_diff, cld_update_param, trn_x, trn_y,
#                            learning_rate, rho, batch_size, epoch, print_per,
#                            weight_decay, dataset_name, sch_step, sch_gamma):
#     n_trn = trn_x.shape[0]
#
#     trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
#                               shuffle=True)
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
#     rho_optimizer = LESAM_D(model.parameters(), optimizer, rho=rho)
#
#     model.train()
#     model = model.to(device)
#
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
#     model.train()
#
#     n_par = get_mdl_params([model_func()]).shape[1]
#
#     for e in range(epoch):
#         # Training
#         epoch_loss = 0
#         trn_gen_iter = trn_gen.__iter__()
#         for i in range(int(np.ceil(n_trn / batch_size))):
#             batch_x, batch_y = trn_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device).reshape(-1).long()
#             rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
#             rho_optimizer.step(cld_update_param)
#
#             # Get linear penalty on the current parameter estimates
#             local_par_list = None
#             for param in model.parameters():
#                 if not isinstance(local_par_list, torch.Tensor):
#                     # Initially nothing to concatenate
#                     local_par_list = param.reshape(-1)
#                 else:
#                     local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
#
#             loss_correct = torch.sum(local_par_list * state_params_diff)
#
#             loss_correct.backward()
#
#             ###
#             # optimizer.zero_grad()
#             # loss.backward()
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
#                                            max_norm=max_norm)  # Clip gradients to prevent exploding
#             optimizer.step()
#             # epoch_loss += loss.item() * list(batch_y.size())[0]
#
#         if (e + 1) % print_per == 0:
#             epoch_loss /= n_trn
#             if weight_decay != None:
#                 # Add L2 loss to complete f_i
#                 params = get_mdl_params([model], n_par)
#                 epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
#
#             print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
#                   % (e + 1, epoch_loss, scheduler.get_lr()[0]))
#
#             model.train()
#         scheduler.step()
#
#     # Freeze model
#     for params in model.parameters():
#         params.requires_grad = False
#     model.eval()
#
#     return model


def train_model_FedALS(model, model_func, cur_cld_model, alpha, global_model_param, parameter_drifts_curr,
                       trn_x, trn_y,
                       learning_rate, batch_size, epoch, kd_T, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_div = DistillKL(kd_T)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=0.1)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()

            y_pred = model(batch_x)
            y_t = cur_cld_model(batch_x)
            loss_k_i = loss_div(y_pred, y_t)
            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - parameter_drifts_curr)) * (
                        local_parameter - (global_model_param - parameter_drifts_curr)))

            loss_correct = loss_cp + loss_k_i
            loss_correct.backward()

            # loss = loss_f_i + loss_cp + loss_cg
            # optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_FedALS_wo_T(model, model_func, alpha, global_model_param, parameter_drifts_curr,
                            trn_x, trn_y,
                            learning_rate, batch_size, epoch, print_per,
                            weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=0.1)

    # model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - parameter_drifts_curr)) * (
                        local_parameter - (global_model_param - parameter_drifts_curr)))

            loss_correct = loss_cp
            loss_correct.backward()

            # loss = loss_f_i + loss_cp + loss_cg
            # optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            # model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_fedvarp_mdl(model, model_func, trn_x, trn_y,
                       learning_rate, batch_size, n_minibatch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_trn / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    count_step = 0
    is_done = False

    step_loss = 0;
    n_data_step = 0
    for e in range(epoch):
        # Training
        ###
        if is_done:
            break
        ###

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            ####
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            ###
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            # loss_algo = torch.sum(local_par_list * state_params_diff)

            # loss = loss_f_i + loss_algo
            loss = loss_f_i

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay) / 2 * np.sum(params * params)

                print("Step %3d, Training Loss: %.4f, LR: %.5f"
                      % (count_step, step_loss, scheduler.get_lr()[0]))
                step_loss = 0;
                n_data_step = 0

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model



def eAM(e,r):
    return math.exp(-e * r)



def train_model_FedAKL(model, model_func, cur_cld_model, alpha, global_model_param, parameter_drifts_curr,
                       trn_x, trn_y,
                       learning_rate, batch_size, epoch, kd_t, aa, bb, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma,feat_s_pre,feat_t_pre,guide_model):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_div = DistillKL(kd_t)
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer1 = torch.optim.SGD(guide_model.parameters(),lr=0.1 * learning_rate,weight_decay=weight_decay)
    model.train()
    model = model.to(device)
    guide_model.train()
    guide_model = guide_model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=sch_step, gamma=sch_gamma)

    n_par = get_mdl_params([model_func()]).shape[1]
    # feat_t_pre_c = [ft.to(device) for (_, ft) in feat_t_pre.items()]
    # feat_s_pre_c = [fs.to(device) for (_, fs) in feat_s_pre.items()]
    # res_fs = {key: feat_s_pre[key] - feat_t_pre[key] for key in feat_s_pre if key in feat_t_pre }

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            res_fs = {key: feat_s_pre[key] - feat_t_pre[key] for key in feat_s_pre if key in feat_t_pre}
            # feat_a = [(fs_pre - ft_pre).repeat(50,1,1,1) for fs_pre, ft_pre in zip(feat_s_pre_c,feat_t_pre_c)]
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            feat_s, y_s = model(batch_x,is_feat=True)
            feat_t, y_t = cur_cld_model(batch_x,is_feat=True)
            feat_t_i,y_t_i = guide_model(batch_x,res_fs)
            # mean_feat_s = {key: torch.mean(feat_s[key],dim=0) for key in feat_s}
            # mean_feat_t = {key: torch.mean(feat_t[key],dim=0) for key in feat_t}
            feat_s_pre = {key: feat_s_pre[key] + feat_s[key].detach() for key in feat_s}
            feat_t_pre = {key: feat_t_pre[key] + feat_t[key].detach() for key in feat_t}

            # print(f'{mean_feat_s}')
            # for key, value in feat_s.items():
            #     feat_s_pre[key] += torch.mean(feat_s[key],dim=0).to('cpu')
            # for key, value in feat_t.items():
            #     feat_t_pre[key] += torch.mean(feat_t[key],dim=0).to('cpu')


            loss_f = loss_fn(y_s, batch_y.reshape(-1).long())

            # loss_kl = loss_div(y_t_i,y_t)
            loss_k_i = loss_div(y_s, y_t_i)
            loss_f_i = [mse_loss(feat_s[key],feat_t_i[key]) for key in feat_s if key in feat_t_i]
            # loss_f_i = [mse_loss(f_s,f_t_i) for f_s, f_t_i in zip(feat_s,feat_t_i)]
            total_loss_f_i = sum(loss_f_i)/len(loss_f_i)
            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - parameter_drifts_curr)) * (
                        local_parameter - (global_model_param - parameter_drifts_curr)))
            if aa >= 1:
                loss = loss_cp +loss_f
            else:
                loss = loss_cp + loss_f + (1 - aa) * total_loss_f_i + (1 - aa) * loss_k_i
            # loss1 = loss_f + loss_k_i + loss_kl + total_loss_f_i
            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            # loss1.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            torch.nn.utils.clip_grad_norm_(parameters=guide_model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            optimizer1.step()
            feat_s_pre = {key: torch.mean(feat_s_pre[key],dim=0) for key in feat_s_pre}
            feat_t_pre = {key: torch.mean(feat_t_pre[key],dim=0) for key in feat_t_pre}





        # feat_t_pre = [ft.to('cpu') for ft in feat_t_pre_c]
        # feat_s_pre = [fs.to('cpu') for fs in feat_s_pre_c]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
        scheduler1.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model,guide_model,feat_s_pre,feat_t_pre



def train_model_FedGkd(model, model_func, cur_cld_model, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                       learning_rate, batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_div = DistillKL(4.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    w_global = get_flat_params(model).clone().detach()
    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            y_t = cur_cld_model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            loss_kl = loss_div(y_pred,y_t)

            # Get linear penalty on the current parameter estimates

            loss = loss_f_i  + 0.1 * loss_kl

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding

            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()


    dp_sigma = 0.95
    dp_clip = 0.2
    clnt = 100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta * min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita = dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

#
# def train_model_FedGkd(model, model_func, cur_cld_model, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
#                        learning_rate, batch_size, epoch, print_per,
#                        weight_decay, dataset_name, sch_step, sch_gamma):
#     n_trn = trn_x.shape[0]
#
#     trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
#                               shuffle=True)
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
#     loss_div = DistillKL(4.0)
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
#
#     model.train();
#     model = model.to(device)
#
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
#     model.train()
#
#     n_par = get_mdl_params([model_func()]).shape[1]
#
#     for e in range(epoch):
#         # Training
#         epoch_loss = 0
#         trn_gen_iter = trn_gen.__iter__()
#         for i in range(int(np.ceil(n_trn / batch_size))):
#             batch_x, batch_y = trn_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#
#             y_pred = model(batch_x)
#             y_t = cur_cld_model(batch_x)
#
#             ## Get f_i estimate
#             loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
#
#             loss_f_i = loss_f_i / list(batch_y.size())[0]
#
#             loss_kl = loss_div(y_pred,y_t)
#
#             # Get linear penalty on the current parameter estimates
#             local_par_list = None
#             for param in model.parameters():
#                 if not isinstance(local_par_list, torch.Tensor):
#                     # Initially nothing to concatenate
#                     local_par_list = param.reshape(-1)
#                 else:
#                     local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
#
#             loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
#
#             loss = loss_f_i + loss_algo + 1 * loss_kl
#
#             ###
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
#                                            max_norm=max_norm)  # Clip gradients to prevent exploding
#
#             # === 原始梯度拉直 ===
#             grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
#             # === 添加噪声 ===
#             noise = torch.randn_like(grads) * 0.1
#             noisy_grad = grads + noise
#             # === 重新写回各层 ===
#             pointer = 0
#             for p in model.parameters():
#                 if p.grad is None:
#                     continue
#                 numel = p.numel()
#                 new_grad = noisy_grad[pointer:pointer + numel].view_as(p)
#                 p.grad.copy_(new_grad)
#                 pointer += numel
#             # === 进行参数更新 ===
#             # optimizer.step()
#             optimizer.step()
#             epoch_loss += loss.item() * list(batch_y.size())[0]
#
#         if (e + 1) % print_per == 0:
#             epoch_loss /= n_trn
#             if weight_decay != None:
#                 # Add L2 loss to complete f_i
#                 params = get_mdl_params([model], n_par)
#                 epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)
#
#             print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
#                   % (e + 1, epoch_loss, scheduler.get_lr()[0]))
#
#             model.train()
#         scheduler.step()
#
#     # Freeze model
#     for params in model.parameters():
#         params.requires_grad = False
#     model.eval()
#
#     return model


def train_model_FedCM(model, model_func, alpha_coef, delta_list, trn_x, trn_y,
                       learning_rate, batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * delta_list)

            loss = alpha_coef * loss_f_i + (1-alpha_coef) *loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model



def train_model_Fedgamma(model, model_func, state_params_diff, trn_x, trn_y,
                         learning_rate, rho, batch_size, n_minibatch,print_per,
                         weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=rho)

    model.train();
    model = model.to(device)
    w_global = get_flat_params(model).clone().detach()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_trn / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    count_step = 0
    is_done = False

    step_loss = 0;
    n_data_step = 0
    for e in range(epoch):
        # Training
        ###
        if is_done:
            break
        ###

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            ####
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            ###
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * state_params_diff)

            loss_algo.backward()

            ###
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()  # Clip gradients to prevent exploding
            step_loss += loss_algo.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay) / 2 * np.sum(params * params)

                print("Step %3d, Training Loss: %.4f, LR: %.5f"
                      % (count_step, step_loss, scheduler.get_lr()[0]))
                step_loss = 0;
                n_data_step = 0

            model.train()
        scheduler.step()


    dp_sigma = 0.95
    dp_clip = 0.2
    clnt = 100
    # 假设本地训练已经完成，得到本地模型参数
    w_local = get_flat_params(model)  # 获取展平的本地参数
    # 1. 计算参数更新量 (delta = w_local - w_global)
    delta = w_local - w_global
    # 2. 裁剪更新量（限制敏感度）
    delta_norm = torch.norm(delta, p=2)
    clipped_delta = delta * min(1.0, dp_clip / delta_norm)
    # 3. 添加高斯噪声
    sita = dp_sigma * dp_clip / clnt
    noise = torch.randn_like(delta) * sita
    noisy_delta = clipped_delta + noise
    # 4. 构造上传参数（保护后的本地更新）
    upload_params = w_global + noisy_delta  # 关键：基于全局参数构造上传值
    set_flat_params(model, upload_params)
    # Freeze model
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_model_MoFedSAM_KL(cur_cld_model,model, model_func, alpha_coef, delta_list, trn_x, trn_y,
                       learning_rate, rho,batch_size, epoch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    kd_T = 4
    loss_div = DistillKL(kd_T)


    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=rho)
    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]

            y_t = cur_cld_model(batch_x)


            rho_optimizer.step(loss_div, y_t)


            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = (1-alpha_coef) * torch.sum(local_par_list * delta_list)


            ###
            loss_algo.backward()

            ###
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss_algo.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model



def train_model_FedA1(model, cur_cld_model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                         learning_rate, rho, batch_size, epoch, print_per,
                         weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_div = DistillKL(4.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay + alpha_coef)
    rho_optimizer = ESAM(model.parameters(), optimizer, rho=rho)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1).long()
            rho_optimizer.paras = [batch_x, batch_y, loss_fn, model]
            rho_optimizer.step()
            y_pred = model(batch_x)
            y_t = cur_cld_model(batch_x)
            loss_k_i = loss_div(y_pred, y_t)

            # y_pred = model(batch_x)
            #
            # ## Get f_i estimate
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            # loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_cp = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            loss_correct = loss_cp + loss_k_i
            loss_correct.backward()

            ###
            # optimizer.zero_grad()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            # epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def train_model_FedFC(model, model_func, alpha, global_model_param, hist_i,
                      trn_x, trn_y,
                      learning_rate, batch_size, epoch, print_per,
                      weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))

            loss = loss_f_i + loss_cp
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model
