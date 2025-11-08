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

    # CIFAR10 with Dirichlet-0.6 non-iid
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

    print('DP-FedDyn (FedDyn + DP-FedAvg protection)')

    epoch = 30
    learning_rate = 0.1
    print_per = 5

    # FedDyn-specific parameter
    lamb = 1e-2

    [fed_mdls_sel, tst_perf_sel] = train_DPFedDyn(
        data_obj=data_obj, act_prob=act_prob,
        learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
        com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
        model_func=model_func, init_model=init_model,
        sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix,
        trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round,
        lamb=lamb)

