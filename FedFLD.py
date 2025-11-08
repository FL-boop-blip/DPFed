from models import model_choose_fn
from utils_general import *
from utils_methods import *
# from utils_methods_FedCM import *
# from utils_methods_FedGamma import *
# from utils_methods_MoFedSAM import *
# from utils_methods_FedA1 import *

# Dataset initialization
data_path = 'Folder/'  # The folder to save Data & Model

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in data_path + "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
#      - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, name, crop_amount = 2000)
#########

n_client = 10
# Generate IID or Dirichlet distribution
# IID
data_obj = DatasetObject(dataset='CIFAR10', n_clnt=n_client, seed=20, rule='Dirichlet',rule_arg=0.3,
                         unbalanced_sgm=0,
                         data_path=data_path)

model_name = 'resnet18'  # Model type

###
# Common hyperparameters

com_amount = 200
save_period = 1
weight_decay = 1e-3
batch_size = 50
# act_prob = 1
act_prob = 0.10
suffix = model_name
lr_decay_per_round = 0.998

out_channel = 10

model_func = lambda: model_choose_fn.choose_model(model_name, num_classes=out_channel)
init_model = model_func()


if not os.path.exists('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
    if not os.path.exists('%sModel/%s/' % (data_path, data_obj.name)):
        print("Create a new directory")
        os.mkdir('%sModel/%s/' % (data_path, data_obj.name))
    torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))

####


out_channel = 10
in_channel = 3
g_model_arch = 'CGeneratorA'
nz = 100

print('FedFLD')
#
epoch = 1
learning_rate = 0.1
alpha_coef = 0.1
print_per = 5
ewc_lambda=0.7
his_lambda=0.5
seed=1024
g_model_func = lambda: model_choose_fn.choose_g_model(g_model_arch, nz=nz, nc=data_obj.channels,
                                                      img_size=data_obj.width, n_cls=out_channel)
init_g_model = g_model_func()

train_FedGAN_common(data_obj=data_obj, act_prob=act_prob,
                                               learning_rate=learning_rate,
                                               batch_size=batch_size, epoch=epoch,
                                               com_amount=com_amount,
                                               print_per=print_per, weight_decay=weight_decay,
                                               model_func=model_func,
                                               init_model=init_model, init_g_model=init_g_model,
                                               sch_step=1, sch_gamma=1,
                                               save_period=save_period, suffix=suffix,
                                               trial=False,
                                               data_path=data_path, rand_seed=seed,
                                               lr_decay_per_round=lr_decay_per_round,
                                               ewc_lambda=ewc_lambda,
                                               his_lambda=his_lambda)
