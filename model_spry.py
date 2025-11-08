import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
# from utils import *
from torch.types import _int, _size
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.grad import conv2d_input, conv2d_weight
import numpy as np
import math
from logging import log
import logging

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


def matrix_drop_th(
        tensor: torch.Tensor,
        select_percentage: float,
) -> torch.Tensor:
    """Remove the elements of the input `tensor` using `TopK` algorithm, where `k` is
    computed using the `select_percentage` parameter and the dimensions of the input
    `tensor`. The input `tensor` is assumed to be a matrix, i.e. a tensor of rank 2.

    Args:
        tensor (torch.Tensor): input tensor to be pruned.
        select_percentage (float): defines the `k` for `TopK`.

    Returns
    -------
        torch.Tensor: output pruned tensor.
        torch.Tensor: value of the threshold used.
    """
    tensor_shape = tensor.shape
    k = int(math.ceil(select_percentage * (tensor_shape[0] * tensor_shape[1])))
    topk = tensor.view(-1).abs().topk(k)
    threshold: torch.Tensor = topk[0][-1]
    index = tensor.abs() >= (threshold)
    index = index.type_as(tensor)
    tensor = tensor * index
    return tensor, threshold


def get_tensor_sparsity(p: torch.tensor) -> float:
    """Count the rate of non-zero parameter in a tensor."""
    tensor = p.data.cpu().numpy()
    nz_count = np.count_nonzero(tensor)
    total_params = np.prod(tensor.shape)
    return 1 - (nz_count / total_params)


def matrix_drop(
        tensor: torch.Tensor,
        select_percentage: float,
) -> torch.Tensor:
    """Same as `matrix_drop_th`, but returns just the pruned tensor.

    Args:
        tensor (torch.Tensor): input tensor to be pruned.
        select_percentage (float): defines the `k` for `TopK`.

    Returns
    -------
        torch.Tensor: output pruned tensor.
    """
    return matrix_drop_th(
        tensor=tensor,
        select_percentage=select_percentage,
    )[0]


def drop_nhwc_send_th(
        tensor: torch.Tensor,
        select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape  # 权重的形状
    # k是要保留的总元素数量, math.ceil向上取整
    k = int(
        math.ceil(
            select_percentage
            * (tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
        )
    )
    # 去除前k个元素（绝对值大小）
    topk = tensor.view(-1).abs().topk(k)
    # 第k大的值，即剪枝的值
    threshold: torch.Tensor = topk[0][-1]
    # 标记所有绝对值大于等于threshold的位置为true
    index = tensor.abs() >= (threshold)
    # 将标记转换为与原始张量相同的数据类型
    index = index.type_as(tensor)
    # 将掩码应用于原张量，梯度友好，允许梯度通过梯度传播回原始张量
    tensor = tensor * index
    return tensor, threshold


def drop_threshold(
        tensor: torch.Tensor,
        threshold: float,
) -> torch.Tensor:
    """Remove the elements of the input tensor that are lower than the input threshold.
    The element-wise pruning decision is performed on the absolute value of the element,
    i.e. the element `x` is pruned if `$|x|<t$` where `t` is the threshold.

    Args:
        tensor (torch.Tensor): input tensor to be pruned.
        threshold (float): threshold used for pruning.

    Returns
    -------
        torch.Tensor: output pruned tensor
    """
    index = tensor.abs() >= threshold
    index = index.type_as(tensor)
    tensor = tensor * index
    return tensor


def drop_structured(
        tensor: torch.Tensor,
        select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    k = int(math.ceil(select_percentage * (tensor_shape[0] * tensor_shape[1])))
    tensor = tensor.reshape(
        tensor_shape[0],
        tensor_shape[1],
        tensor_shape[2] * tensor_shape[3],
    )

    new_tensor_shape = tensor.shape
    value = torch.sum(tensor.abs(), 2)
    topk = value.view(-1).abs().topk(k)
    interleaved: torch.Tensor = topk[0][-1]
    index = value.abs() >= (interleaved)
    index = (
        index.repeat_interleave(tensor_shape[2] * tensor_shape[3])
        .type_as(tensor)
        .reshape(new_tensor_shape)
    )

    tensor = tensor * index
    tensor = tensor.reshape(tensor_shape)
    return tensor


def drop_structured_filter(
        tensor: torch.Tensor,
        select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    k = int(math.ceil(select_percentage * (tensor_shape[0])))
    tensor = tensor.reshape(
        tensor_shape[0],
        tensor_shape[1] * tensor_shape[2] * tensor_shape[3],
    )

    new_tensor_shape = tensor.shape
    value = torch.sum(tensor.abs(), 1)
    topk = value.view(-1).abs().topk(k)
    interleaved: torch.Tensor = topk[0][-1]
    index = value.abs() >= (interleaved)
    index = (
        index.repeat_interleave(tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
        .type_as(tensor)
        .reshape(new_tensor_shape)
    )

    tensor = tensor * index
    tensor = tensor.reshape(tensor_shape)
    return tensor


def calculate_fan_in(tensor: torch.Tensor) -> float:
    # 计算权重张量的输入维度
    """Calculate fan in.

    Modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    """
    min_fan_in = 2
    dimensions = tensor.dim()
    if dimensions < min_fan_in:
        raise ValueError(
            "Fan in can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if dimensions > min_fan_in:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    return float(fan_in)


class SpectralNormHandler:
    def __init__(self, epsilon: float = 1e-12, num_iterations: int = 1):
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.exponent_range = 1.0
        self.cached_exponents: dict = {}
        self.sigma = 0.0

    def _compute_spectral_norm(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute the spectral normalization for a given weight tensor."""
        weight_detach = weight.detach()
        weight_abs = weight_detach.abs()
        weight_mat = weight_abs.view(weight_abs.size(0), -1)

        u = torch.randn(weight_mat.size(0), 1, device=weight.device)
        v = torch.randn(weight_mat.size(1), 1, device=weight.device)

        for _ in range(self.num_iterations):
            v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0)
            u = F.normalize(torch.matmul(weight_mat, v), dim=0)

        sigma = torch.matmul(u.t(), torch.matmul(weight_mat, v))
        self.sigma = torch.clamp(sigma, min=self.epsilon)

        return weight_abs / self.sigma

    def _get_cache_key(self, tensor: torch.Tensor) -> str:
        """Generate a unique key for the tensor based on its properties."""
        return f"{tensor.shape}_{tensor.device}_{tensor.dtype}"

    def compute_weight_update(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute the updated weight with cached exponents."""
        cache_key = self._get_cache_key(weight)

        # Get or compute the exponent
        if cache_key not in self.cached_exponents:
            # Compute the normalized weight
            weight_normalized = self._compute_spectral_norm(weight)

            # comute the weight_normalized average value
            weight_normalized_avg = torch.mean(weight_normalized)
            # compute the average value of the non zero weight
            # weight_normalized_avg = torch.mean(
            #     weight_normalized[weight_normalized != 0]
            # )

            # Compute the exponent
            # exponent = 1 + (self.exponent_range * weight_normalized.view_as(weight))
            # exponent = torch.clamp(exponent, max=10)  # Prevent overflow
            # exponent = 1 + weight_normalized_avg

            # add a check if exponent is nan
            if torch.isnan(weight_normalized_avg):
                log(
                    logging.INFO,
                    "Exponent is nan. Number of non zero values:"
                    f" {torch.sum(weight_normalized != 0)}",
                )
                exponent = 1.0
            else:
                weight_normalized_avg = round(weight_normalized_avg.item(), 8)
                exponent = 1.0 + weight_normalized_avg

            # Save the exponent in cache
            self.cached_exponents[cache_key] = exponent

        # Use cached exponent
        exponent = self.cached_exponents[cache_key]

        # Compute final weight update
        sign_weight = torch.sign(weight)
        weight_abs = weight.abs()
        weight_updated = sign_weight * torch.pow(weight_abs, exponent)
        # normalize the weight to the original sigma
        # weight_updated = weight_updated * (
        #     weight_updated / torch.pow(self.sigma, 1 + self.exponent_range)
        # )

        return weight_updated

    def clear_cache(self):
        """Clear the cached exponents."""
        self.cached_exponents = {}


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResEmoteNet(nn.Module):
    def __init__(self):
        super(ResEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(256)

        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.se(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x


class swat_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, sparsity):

        if input.dim() == 2 and bias is not None:
            # The fused op is marginally faster
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias

        if sparsity != 0.0:
            sparse_input = matrix_drop(input, 1 - sparsity)
        else:
            sparse_input = input

        ctx.save_for_backward(sparse_input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_input, sparse_weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(sparse_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(sparse_input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class SWATLinear(nn.Module):
    """Powerpropagation Linear module."""

    def __init__(
            self,
            alpha: float,
            in_features: int,
            out_features: int,
            bias: bool = True,
            sparsity: float = 0.3,
    ):
        super(SWATLinear, self).__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None
        self.sparsity = sparsity
        if alpha == 1.0:
            self.weight_sparsity = sparsity
        else:
            self.weight_sparsity = 0.0

    def __repr__(self):
        return (
            f"SWATLinear(alpha={self.alpha}, in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b},"
            f" sparsity={self.sparsity})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def _call_swat_linear(self, input, weight) -> torch.Tensor:
        if self.training:
            sparsity = self.sparsity
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0
        return swat_linear.apply(input, weight, self.bias, sparsity)

    def forward(self, input):
        if (
                self.training
                and (self.weight_sparsity != 0.0 or self.alpha == 1.0)
                and self.sparsity != 0.0
        ):
            self.weight.data = matrix_drop(self.weight, 1 - self.weight_sparsity)

        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            powerprop_weight = self.weight
        else:
            powerprop_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )

        # Perform SWAT forward pass
        output = self._call_swat_linear(input, powerprop_weight)

        # Return the output
        return output


class swat_conv2d(Function):
    @staticmethod
    def forward(
            ctx,
            input,
            weight,
            bias,
            sparsity,
            in_threshold,
            stride,
            padding,
            dilation,
            groups,
    ):
        # 确保张量在内存中连续存储，提升计算效率
        input = input.contiguous()
        # 进行卷积计算
        output = F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        # 将input输入稀疏化处理
        if sparsity != 0.0:
            if in_threshold < 0.0:
                # in_threshold 为input中第 （1 - sparsity）* number of input 个元素的阈值
                sparse_input, in_threshold_tensor = drop_nhwc_send_th(
                    input, 1 - sparsity
                )
                in_threshold = in_threshold_tensor.item()
            else:
                # 如果in_threshold已经存在，则直接使用该阈值进行稀疏化（直接把阈值写死，基本不会用）
                sparse_input = drop_threshold(input, in_threshold)

        else:
            sparse_input = input
        # 保存反向传播所需的数据
        # 将卷积配置（步长、填充等）存入ctx.conf
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        # 保存稀疏化后的输入、卷积核的权重、偏置
        ctx.save_for_backward(sparse_input, weight, bias)
        # 返回结果和input的裁剪阈值
        return output, in_threshold

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    # 限制该函数仅能被自动微分系统调用一次，防止搞阶梯度计算
    # def backward(ctx, grad_output, grad_wt_th, grad_in_th):
    # grad_output:损失函数对前向传播（稀疏卷积）输出output、sparse_input的梯度
    def backward(ctx, grad_output, grad_in_th):
        # grad_output = grad_output.contiguous()
        return convolution_backward(ctx, grad_output)


class SWATConv2D(nn.Module):
    """Powerpropagation Conv2D module."""

    def __init__(
            self,
            alpha: float,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: Union[_size, _int] = 1,
            padding: Union[_size, _int] = 1,
            dilation: Union[_size, _int] = 1,
            groups: _int = 1,
            bias: bool = False,
            sparsity: float = 0.3,
            pruning_type: str = "unstructured",
            warm_up: int = 0,
            period: int = 1,
    ):
        super(SWATConv2D, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.b = bias
        # 权重和偏置初始化
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.sparsity = sparsity
        if alpha == 1.0:
            self.weight_sparsity = sparsity
        else:
            self.weight_sparsity = 0.0
        self.pruning_type = pruning_type
        self.warmup = warm_up
        self.period = period
        self.wt_threshold = -1.0
        self.in_threshold = -1.0
        self.epoch = 0
        self.batch_idx = 0

    def __repr__(self):
        return (
            f"SWATConv2D(alpha={self.alpha}, in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups},"
            f" sparsity={self.sparsity}, pruning_type={self.pruning_type},"
            f" warm_up={self.warmup}, period={self.period})"
        )

    # weights re-parametrization
    def get_weight(self):
        weight = self.weight.detach()
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def _call_swat_conv2d(self, input, weight) -> torch.Tensor:

        # To compute the in_threshold evry round just put it to -1.0
        # if self.epoch >= self.warmup:
        #     if self.batch_idx % self.period == 0:
        #         self.in_threshold = torch.tensor(-1.0)

        if self.training:
            sparsity = self.sparsity
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0

        output, in_threshold = swat_conv2d.apply(
            input,
            weight,
            self.bias,
            sparsity,
            self.in_threshold,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # update self.in_threshold
        if sparsity != 0.0:
            # otherwise, it is not updated
            self.in_threshold = in_threshold

        return output

    def forward(self, input):
        # sparsify the weights
        if (
                self.training
                and (self.weight_sparsity != 0.0 or self.alpha == 1.0)
                and self.sparsity != 0.0
        ):
            top_k = 1 - self.weight_sparsity
            # 对模型权重进行非结构化剪枝，得到稀疏化的权重
            if self.pruning_type == "unstructured":
                if self.wt_threshold < 0.0:
                    # Here you have to compute the threshold
                    self.weight.data, wt_threshold_tensor = drop_nhwc_send_th(
                        self.weight, top_k
                    )
                    self.wt_threshold = wt_threshold_tensor.item()
                else:
                    # You already have the threshold
                    self.weight.data = drop_threshold(self.weight, self.wt_threshold)
            elif self.pruning_type == "structured_channel":
                self.weight.data = drop_structured(self.weight, top_k)
            elif self.pruning_type == "structured_filter":
                self.weight.data = drop_structured_filter(self.weight, top_k)
            else:
                assert 0, "Illegal Pruning Type"

        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        # weight re-parametrization to get $\theta_^t$
        if self.alpha != 1.0:
            powerprop_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        else:
            powerprop_weight = self.weight

            # Perform the forward pass
        # 这里的powerprop_weight是经过掩码和重参数化的权重
        # input为进行处理 在 _call_swat_conv2d 函数中进行处理
        output = self._call_swat_conv2d(
            input,
            powerprop_weight,
        )

        # Return the output
        return output  # a^t


class SparsyFed_no_act_linear(nn.Module):
    """SparsyFed (no activation pruning) Linear module."""

    def __init__(
            self,
            alpha: float,
            sparsity: float,
            in_features: int,
            out_features: int,
            bias: bool = True,
    ):
        super(SparsyFed_no_act_linear, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None
        self.spectral_norm_handler = SpectralNormHandler()

    def __repr__(self):
        return (
            f"SparsyFed_no_act_linear(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b})"
        )

    def get_weight(self):
        weight = self.weight.detach()
        if self.alpha == 1.0:
            return weight
        elif self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weight)
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            weight = self.weight
        elif self.alpha < 0:
            weight = self.spectral_norm_handler.compute_weight_update(self.weight)
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        # Apply a mask, if given
        if mask is not None:
            weight *= mask
        # Compute the linear forward pass usign the re-parametrised weight
        return F.linear(input=inputs, weight=weight, bias=self.bias)


class SparsyFed_no_act_Conv2D(nn.Module):
    """SparsyFed (no activation pruning) Conv2D module."""

    def __init__(
            self,
            alpha: float,
            sparsity: float,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: Union[_size, _int] = 1,
            padding: Union[_size, _int] = 1,
            dilation: Union[_size, _int] = 1,
            groups: _int = 1,
            bias: bool = False,
    ):
        super(SparsyFed_no_act_Conv2D, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.b = bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.spectral_norm_handler = SpectralNormHandler()

    def __repr__(self):
        return (
            f"SparsyFed_no_act_Conv2D(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        elif self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weights)
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            weight = self.weight
        elif self.alpha < 0:
            weight = self.spectral_norm_handler.compute_weight_update(self.weight)
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        # Apply a mask, if given
        if mask is not None:
            weight *= mask
        # Compute the conv2d forward pass usign the re-parametrised weight
        return F.conv2d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SparsyFed_no_act_Conv1D(nn.Module):
    """SparsyFed (no activation pruning) Conv1D module."""

    def __init__(
            self,
            alpha: float,
            sparsity: float,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: Union[_size, _int] = 1,
            padding: Union[_size, _int] = 1,
            dilation: Union[_size, _int] = 1,
            groups: _int = 1,
            bias: bool = False,
    ):
        super(SparsyFed_no_act_Conv1D, self).__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.spectral_norm_handler = SpectralNormHandler()

    def __repr__(self):
        return (
            f"SparsyFed_no_act_Conv1D(alpha={self.alpha}, sparsity={self.sparsity},"
            f" in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            weight = self.weight
        else:
            weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        # Apply a mask, if given
        if mask is not None:
            weight *= mask

        return F.conv1d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


torch.autograd.set_detect_anomaly(True)


def convolution_backward(
        ctx,
        grad_output,
):
    sparse_input, sparse_weight, bias = ctx.saved_tensors
    conf = ctx.conf
    input_grad = (
        weight_grad
    ) = (
        bias_grad
    ) = (
        sparsity_grad
    ) = (
        grad_in_th
    ) = grad_wt_th = stride_grad = padding_grad = dilation_grad = groups_grad = None

    # Compute gradient w.r.t. input
    if ctx.needs_input_grad[0]:
        input_grad = conv2d_input(
            sparse_input.shape,
            sparse_weight,
            grad_output,
            conf["stride"],
            conf["padding"],
            conf["dilation"],
            conf["groups"],
        )

    # Compute gradient w.r.t. weight
    if ctx.needs_input_grad[1]:
        weight_grad = conv2d_weight(
            sparse_input,
            sparse_weight.shape,
            grad_output,
            conf["stride"],
            conf["padding"],
            conf["dilation"],
            conf["groups"],
        )

    # Compute gradient w.r.t. bias (works for every Conv2d shape)
    if bias is not None and ctx.needs_input_grad[2]:
        bias_grad = grad_output.sum(dim=(0, 2, 3))

    return (
        input_grad,
        weight_grad,
        bias_grad,
        sparsity_grad,
        grad_in_th,
        grad_wt_th,
        stride_grad,
        padding_grad,
        dilation_grad,
        groups_grad,
    )


class sparsyfed_linear(Function):
    threshold = 1e-7

    @staticmethod
    def forward(ctx, input, weight, bias, sparsity):

        if input.dim() == 2 and bias is not None:
            # The fused op is marginally faster
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias

        topk = max(1 - sparsity, sparsyfed_linear.threshold)

        sparse_input = matrix_drop(input, topk)

        ctx.save_for_backward(sparse_input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_input, sparse_weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(sparse_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(sparse_input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class SparsyFedLinear(nn.Module):
    """Powerpropagation Linear module."""

    def __init__(
            self,
            alpha: float,
            in_features: int,
            out_features: int,
            bias: bool = True,
            sparsity: float = 0.3,
    ):
        super(SparsyFedLinear, self).__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None
        self.spectral_norm_handler = SpectralNormHandler()
        self.sparsity = sparsity

    def __repr__(self):
        return (
            f"SparsyFedLinear(alpha={self.alpha}, in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b},"
            f" sparsity={self.sparsity})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        elif self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weights)
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def _call_sparsyfed_linear(self, input, weight) -> torch.Tensor:
        if self.training:
            sparsity = get_tensor_sparsity(weight)
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0
        return sparsyfed_linear.apply(input, weight, self.bias, sparsity)

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            sparsyfed_weight = self.weight
        elif self.alpha < 0:
            sparsyfed_weight = self.spectral_norm_handler.compute_weight_update(
                self.weight
            )
        else:
            sparsyfed_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )

        output = self._call_sparsyfed_linear(input, sparsyfed_weight)

        # Return the output
        return output


class sparsyfed_conv2d(Function):
    threshold = 1e-7

    # threshold = 0

    @staticmethod
    def forward(
            ctx,
            input,
            weight,
            bias,
            sparsity,
            in_threshold,
            stride,
            padding,
            dilation,
            groups,
    ):
        # Ensure input tensor is contiguous
        input = input.contiguous()

        output = F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        topk = max(1 - sparsity, sparsyfed_conv2d.threshold)

        sparse_input = matrix_drop(input, topk)
        if in_threshold < 0.0:
            sparse_input, in_threshold_tensor = drop_nhwc_send_th(input, topk)
            in_threshold = in_threshold_tensor.item()
        else:
            sparse_input = drop_threshold(input, in_threshold)  # \hat{a}_{t,l} in_threshold = 1e-7

        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

        ctx.save_for_backward(sparse_input, weight, bias)

        return output, in_threshold

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    # def backward(ctx, grad_output, grad_wt_th, grad_in_th):
    def backward(ctx, grad_output, grad_in_th):
        grad_output = grad_output.contiguous()
        return convolution_backward(ctx, grad_output)


class SparsyFedConv2D(nn.Module):
    """Powerpropagation Conv2D module."""

    def __init__(
            self,
            alpha: float,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: Union[_size, _int] = 1,
            padding: Union[_size, _int] = 1,
            dilation: Union[_size, _int] = 1,
            groups: _int = 1,
            bias: bool = False,
            sparsity: float = 0.3,
            pruning_type: str = "unstructured",
            warm_up: int = 0,
            period: int = 1,
    ):
        super(SparsyFedConv2D, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.b = bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.sparsity = sparsity
        self.pruning_type = pruning_type
        self.warmup = warm_up
        self.period = period
        self.wt_threshold = -1.0
        self.in_threshold = -1.0
        self.epoch = 0
        self.batch_idx = 0
        self.spectral_norm_handler = SpectralNormHandler()

    def __repr__(self):
        return (
            f"SparsyFedConv2D(alpha={self.alpha}, in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups},"
            f" sparsity={self.sparsity}, pruning_type={self.pruning_type},"
            f" warm_up={self.warmup}, period={self.period})"
        )

    def get_weight(self):
        weight = self.weight.detach()
        if self.alpha == 1.0:
            return weight
        if self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weight)
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def _call_sparsyfed_conv2d(self, input, weight) -> torch.Tensor:

        if self.training:
            # for the activation the sparsity used is proportional to the weight sparsity
            # 计算权重稀疏性
            sparsity = get_tensor_sparsity(weight)  # s_{t,l}
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0

        output, in_threshold = sparsyfed_conv2d.apply(
            input,
            weight,
            self.bias,
            sparsity,
            self.in_threshold,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # update self.in_threshold
        if sparsity != 0.0:
            # otherwise, it is not updated
            self.in_threshold = in_threshold

        return output  # a_{t,l}

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            sparsyfed_weight = self.weight
        elif self.alpha < 0:
            sparsyfed_weight = self.spectral_norm_handler.compute_weight_update(
                self.weight
            )
        else:
            sparsyfed_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )
        # print("dynamic sparsity:", self.sparsity)
        # Perform the forward pass
        output = self._call_sparsyfed_conv2d(
            input,
            sparsyfed_weight,
        )

        # Return the output
        return output


def replace_layer_with_sparsyfed(
        module: nn.Module,
        name: str = "Model",
        alpha: float = 1.0,
        sparsity: float = 0.0,
        pruning_type: str = "unstructured",
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the SWAT versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            new_conv = SparsyFedConv2D(
                alpha=alpha,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                bias=target_attr.bias is not None,
                padding=target_attr.padding,
                stride=target_attr.stride,
                sparsity=sparsity,
                pruning_type=pruning_type,
                warm_up=0,
                period=1,
            )
            setattr(module, attr_str, new_conv)
        if type(target_attr) == nn.Linear:
            new_conv = SparsyFedLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            setattr(module, attr_str, new_conv)

    for model, immediate_child_module in module.named_children():
        replace_layer_with_sparsyfed(immediate_child_module, model, alpha, sparsity)


def init_weights(module: nn.Module) -> None:
    """Initialise standard and custom layers in the input module."""
    if isinstance(
            module,
            SparsyFed_no_act_linear
            | SparsyFed_no_act_Conv2D
            | SparsyFed_no_act_Conv1D
            | SparsyFedLinear
            | SparsyFedConv2D
            | SWATConv2D
            | SWATLinear
            | nn.Linear
            | nn.Conv2d
            | nn.Conv1d,
    ):
        # Your code here
        fan_in = calculate_fan_in(module.weight.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978

        std = np.sqrt(1.0 / fan_in) / distribution_stddev
        # 阶段范围
        a, b = -2.0 * std, 2.0 * std

        # 对权重张量进行截断正态分布初始化，避免梯度消失/爆炸
        u = nn.init.trunc_normal_(module.weight.data, std=std, a=a, b=b)
        if (
                isinstance(
                    module,
                    SparsyFed_no_act_linear
                    | SparsyFed_no_act_Conv2D
                    | SparsyFedLinear
                    | SparsyFedConv2D,
                )
                and module.alpha > 1
        ):
            u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)
            # weight = sign(weight) * |weight|^(1/alpha) 这里的 1/alpha对应论文中的beta

        module.weight.data = u
        if module.bias is not None:
            module.bias.data.zero_()


if __name__ == "__main__":
    model = ResEmoteNet()
    print(model)

    # Test the model with a random input
    x = torch.randn(2, 3, 64, 64)  # Batch size of 1, 3 channels, 224x224 image
    output = model(x)
    print(output.shape)  # Should be [1, 7] for the 7 classes