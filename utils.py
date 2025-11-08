import torch
import torch.nn.functional as F
from logging import log
import logging
from model_spry import SparsyFed_no_act_Conv1D, SparsyFed_no_act_Conv2D, SparsyFed_no_act_linear, SparsyFedConv2D, \
    SparsyFedLinear, SWATConv2D, SWATLinear
import torch.nn as nn
import numpy as np
import math


def get_mdl_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)


def set_clnt_from_params(device, model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)


def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)


def set_client_from_params(device, model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)


def get_params_list_with_shape(model, param_list):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape))
    return vec_with_shape


def get_params_to_prune(model, first_layer=False):
    """
    Get parameters to prune in the model.
    """
    params_to_prune = []
    first_layer = first_layer

    def add_immediate_child(
            module: nn.Module,
            name: str,
    ) -> None:
        nonlocal first_layer
        if (
                type(module) == SparsyFed_no_act_Conv2D
                or type(module) == SparsyFed_no_act_Conv1D
                or type(module) == SparsyFed_no_act_linear
                or type(module) == SparsyFedConv2D
                or type(module) == SparsyFedLinear
                or type(module) == SWATConv2D
                or type(module) == SWATLinear
                or type(module) == nn.Conv2d
                or type(module) == nn.Conv1d
                or type(module) == nn.Linear
        ):
            if first_layer:
                first_layer = False
            else:
                params_to_prune.append((module, "weight", name))
        for _name, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module, _name)

    add_immediate_child(model, "Net")

    return params_to_prune