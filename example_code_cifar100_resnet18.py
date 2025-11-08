import torch

from utils_general import *
from utils_methods import *

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
#     - Change the datase                                                                                             t argument from CIFAR10 to emnist.
########
#      - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, name, crop_amount = 2000)
#########

if __name__ == '__main__':

    n_client = 100
    # Generate IID or Dirichlet distribution
    # IID
    data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, seed=20, rule='Dirichlet', rule_arg=0.3,
                             unbalanced_sgm=0,
                             data_path=data_path)
    # unbalanced
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)

    # Dirichlet (0.6)
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
    # Dirichlet (0.3)
    # data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)

    # model_name = 'cifar10_LeNet'  # Model type
    model_name = 'Resnet18'
    # model_name_1 = 'cifar10_LeNet'

    ###
    # Common hyperparameters

    com_amount = 1000
    save_period = 1000
    weight_decay = 1e-3
    batch_size = 50
    # act_prob = 1
    act_prob = 0.10
    suffix = model_name
    lr_decay_per_round = 0.998

    # Model function
    model_func = lambda: client_model(model_name)
    init_model = model_func()
    dummy_data = torch.randn(50,3,32,32)
    # feats, y = init_model(dummy_data,is_feat = True)
    tuning_modules = nn.ModuleDict({
        'layer1': TuningModule(64, 64),
        'layer2': TuningModule(128, 128),
        'layer3': TuningModule(256, 256),
        'layer4': TuningModule(512, 512)
    })

    features_map = {}
    def hook_fn(name):
        def hook(module, input, output):
            features_map[name] = output
        return hook

    init_model.layer1.register_forward_hook(hook_fn('layer1'))
    init_model.layer2.register_forward_hook(hook_fn('layer2'))
    init_model.layer3.register_forward_hook(hook_fn('layer3'))
    init_model.layer4.register_forward_hook(hook_fn('layer4'))

    for name,layer in init_model.named_parameters():
        print(f'{name}')
    with torch.no_grad():
        _ = init_model(dummy_data)
    feat_s_pre = {}
    feat_t_pre = {}
    for name, feature in features_map.items():
        feat_s_pre[name] = torch.mean(torch.zeros_like(feature),dim=0).unsqueeze(0).to(device)
        feat_t_pre[name] = torch.mean(torch.zeros_like(feature),dim=0).unsqueeze(0).to(device)

    init_guide_model = ModelWithTuning(init_model, tuning_modules)


    # feat_s_pre = {}
    # feat_s_pre = [torch.zeros_like(f_s) for f_s in feats]
    # feat_t_pre = [torch.zeros_like(f_s) for f_s in feats]

    # Initalise the model for all methods with a random seed or load it from a saved initial model
    torch.manual_seed(37)
    if not os.path.exists('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
        if not os.path.exists('%sModel/%s/' % (data_path, data_obj.name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' % (data_path, data_obj.name))
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))

    print('Fedsgam')

    epoch = 5
    alpha_coef = 0.1
    learning_rate = 0.1
    print_per = epoch // 2
    rho = 0.1
    #
    [avg_cld_mdls, tst_cur_cld_perf] = train_FedSGAM(data_obj=data_obj, act_prob=act_prob,
                                                     learning_rate=learning_rate,
                                                     batch_size=batch_size, epoch=epoch,
                                                     com_amount=com_amount, print_per=print_per,
                                                     weight_decay=weight_decay,
                                                     model_func=model_func, init_model=init_model,
                                                     alpha_coef=alpha_coef, rho=rho,
                                                     sch_step=1, sch_gamma=1,
                                                     save_period=save_period,
                                                     suffix=suffix, trial=False,
                                                     data_path=data_path,
                                                     lr_decay_per_round=lr_decay_per_round)
    # #