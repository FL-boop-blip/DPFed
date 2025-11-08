import torch

from utils_general import *
from utils_methods import *
from test import *
from utils_models import *

# Dataset initialization
data_path = 'Folder/'  # The folder to save Data & Model

if __name__ == '__main__':

    check_available_gpus()
    n_client = 100

    # CIFAR10 with Dirichlet-0.6 non-iid (same as DPFedAvg example)
    data_obj = DatasetObject(dataset='CIFAR10', n_clnt=n_client, seed=20, rule='Dirichlet', rule_arg=0.6,
                             data_path=data_path)

    model_name = 'cifar10_LeNet'

    # Common hyperparameters
    com_amount = 300
    save_period = 400
    weight_decay = 1e-3
    batch_size = 50
    act_prob = 0.10
    suffix = model_name
    lr_decay_per_round = 0.998

    # Model function
    model_func = lambda: client_model(model_name)
    init_model = model_func()

    # Initialise / load init model
    torch.manual_seed(37)
    if not os.path.exists('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' % (data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' % (data_path, data_obj.name))
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
    else:
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))

    print('DP-FedGloss (ESAM + DP-FedAvg style protection)')

    epoch = 30
    learning_rate = 0.1
    print_per = 5

    # FedGloss-specific parameters
    lamb = 1e-3      # linear correction strength
    rho_l = 0.1      # max server perturbation strength
    T_s = 100        # warmup rounds for rho
    esam_rho = 0.5   # local ESAM radius

    [fed_mdls_sel, tst_perf_sel] = train_DPFedGloss(
        data_obj=data_obj, act_prob=act_prob,
        learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
        com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
        model_func=model_func, init_model=init_model,
        sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
        trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round,
        lamb=lamb, rho_l=rho_l, T_s=T_s, esam_rho=esam_rho)

