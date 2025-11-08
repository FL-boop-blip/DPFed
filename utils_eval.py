import math

import numpy as np
import torch

from optimizer.GAM import GAM
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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_acc(tst_loader,model):
    acc_overall = 0
    n_tst = 0
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for images, labels in tst_loader:
            images, labels = images.to(device), labels.to(device)
            y_pred = model(images)
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            n_tst += labels.shape[0]
            labels = labels.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == labels)
            acc_overall += batch_correct * 100
    return acc_overall / n_tst



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                         std=[0.247, 0.243, 0.262])])
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.491],
                                                          std=[0.247])
                                     ])
    transform2 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.482],
                                                          std=[0.243])
                                     ])
    transform3 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.447],
                                                          std=[0.262])
                                     ])
    tstset = torchvision.datasets.CIFAR10(root='Folder/Data/Raw',
                                          train=False, download=True, transform=transform)
    tstset1 = torchvision.datasets.CIFAR10(root='Folder/Data/Raw',
                                          train=False, download=True,transform=transform1)
    tstset2 = torchvision.datasets.CIFAR10(root='Folder/Data/Raw',
                                          train=False, download=True,transform=transform2)
    tstset3 = torchvision.datasets.CIFAR10(root='Folder/Data/Raw',
                                          train=False, download=True,transform=transform3)
    tst_load = torch.utils.data.DataLoader(tstset, batch_size=2000, shuffle=True)
    tst_load1 = torch.utils.data.DataLoader(tstset1, batch_size=2000, shuffle=True)
    tst_load2 = torch.utils.data.DataLoader(tstset1, batch_size=2000, shuffle=True)
    tst_load3 = torch.utils.data.DataLoader(tstset1, batch_size=2000, shuffle=True)
    model_name = 'Resnet18'
    model_path = 'Folder/Model/CIFAR10_100_20_Dirichlet_0.300/FedGKD_Resnet18_S200_F0.100000_Lr0.100000_1_1.000000_B50_E5_W0.001000_a0.100000_seed0_lrdecay0.999800/cld_avg_1000com.pt'
    model_func = lambda: client_model(model_name)
    trained_model = model_func()
    trained_model.load_state_dict(torch.load(model_path,map_location=device))
    num_tests = 4
    accuracies = []


    accuracy = get_acc(tst_load, trained_model)
    accuracy1 = get_acc(tst_load1, trained_model)
    accuracy2 = get_acc(tst_load2, trained_model)
    accuracy3 = get_acc(tst_load3, trained_model)
    accuracies.append(accuracy)
    accuracies.append(accuracy1)
    accuracies.append(accuracy2)
    accuracies.append(accuracy3)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f'Accuracy over {num_tests} runs: {mean_accuracy:.2f} ± {std_accuracy:.2f}%')