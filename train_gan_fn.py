import numpy as np
import torch
import os
import torch.nn as nn
from models import model_choose_fn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ewc(fisher_information, weight, weight_old, lambda_):
    return lambda_ / 2.0 * torch.sum(torch.tensor(fisher_information) * (weight - weight_old) ** 2.0)


def get_class_number(data_obj):
    clnt_cls_num = np.zeros((data_obj.n_clnt, data_obj.n_cls))
    clnt_y = data_obj.clnt_y
    for i in range(data_obj.n_clnt):
        for j in clnt_y[i]:
            clnt_cls_num[i, j] += 1
    return clnt_cls_num

def generate_labels(number, cls_num):
    labels = np.arange(number)
    proportions = cls_num / cls_num.sum()
    proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
    labels_split = np.split(labels, proportions)
    for i in range(len(labels_split)):
        labels_split[i].fill(i)
    labels = np.concatenate(labels_split)
    np.random.shuffle(labels)
    return labels.astype(int)

def get_batch_weight(labels, cls_clnt_weight):
    bs = labels.size
    num_clients = cls_clnt_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    batch_weight[np.arange(bs), :] = cls_clnt_weight[labels, :]
    return batch_weight

def train_generator(z, y_onehot, labels,
                                generator, student, teacher,
                                weight, num_clients, i, suffix,
                                data_obj, data_path, ewc_lambda, train_fedgen_feature=False):
    g_model_func = lambda: model_choose_fn.choose_g_model('CGeneratorA', nz=100, nc=3,
                                                          img_size=32, n_cls=10)
    if i>1:
        lambda_cls = 1.0
        lambda_div = 1.0
        cls_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        diversity_criterion = DiversityLoss(metric='l1').to(device)

        y = torch.Tensor(labels).long().to(device)
        fake = generator(z, y_onehot)
        t_logit = teacher(fake, train_fedgen_feature)
        s_logit = student(fake, train_fedgen_feature)
        loss_tran = - torch.mean(
            torch.mean(torch.abs(s_logit - t_logit.detach()), dim=1) * weight)

        loss_cls = torch.mean(cls_criterion(t_logit, y) * weight.squeeze())

        loss_div = diversity_criterion(z.view(z.shape[0], -1), fake)

        init_g_modelpre = g_model_func().to(device)
        init_g_modelcopy = g_model_func().to(device)
        init_g_modelpre.load_state_dict(torch.load('%sModel/%s/%s/%dcom_G.pt'
                                                   % (data_path, data_obj.name, suffix, (i - 1))))
        init_g_modelcopy.load_state_dict(torch.load('%sModel/%s/%s/%dcom_G.pt'
                                                    % (data_path, data_obj.name, suffix, (i))))
        gradients = [param.flatten().detach().cpu().numpy() for param in generator.parameters()]
        fisher_info = []
        fisher_info.append(np.square(np.concatenate(gradients)) / 10000)
        fisher_info = np.mean(fisher_info, axis=0)
        fisher_info = torch.from_numpy(fisher_info)
        fisher_info = fisher_info.to(device)
        prev_weights = torch.cat([param.view(-1) for param in init_g_modelpre.parameters()])
        current_weights = torch.cat([param2.view(-1) for param2 in init_g_modelcopy.parameters()])
        ewc_value = ewc(fisher_info, current_weights, prev_weights, 0.9)
        loss = loss_tran + lambda_cls * loss_cls + lambda_div * loss_div / num_clients + ewc_lambda * ewc_value
        loss.backward()
        return loss, loss_tran, loss_cls, loss_div, ewc_value
    else:
        lambda_cls = 1.0
        lambda_div = 1.0
        ewc_value = 0.0
        cls_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        diversity_criterion = DiversityLoss(metric='l1').to(device)

        y = torch.Tensor(labels).long().to(device)

        fake = generator(z, y_onehot)
        t_logit = teacher(fake, train_fedgen_feature)
        s_logit = student(fake, train_fedgen_feature)
        loss_tran = - torch.mean(
            torch.mean(torch.abs(s_logit - t_logit.detach()), dim=1) * weight)

        loss_cls = torch.mean(cls_criterion(t_logit, y) * weight.squeeze())

        loss_div = diversity_criterion(z.view(z.shape[0],-1), fake)
        loss = loss_tran + lambda_cls * loss_cls + lambda_div * loss_div / num_clients
        loss.backward()
        return loss, loss_tran, loss_cls, loss_div, ewc_value

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
