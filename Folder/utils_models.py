import torch
from pyexpat import features

from utils_libs import *
import torchvision.models as models


def replace_bn_with_gn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # 获取 BN 层的参数
            num_features = module.num_features
            # 选择合适的组数，使得 num_features 能被 num_groups 整除
            num_groups = min(32, num_features)  # 选择 32 或更小的组数
            while num_features % num_groups != 0:
                num_groups -= 1
            # 创建 GN 层
            gn_layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
            # 替换 BN 层为 GN 层
            setattr(model, name, gn_layer)
        else:
            # 递归替换子模块中的 BN 层
            replace_bn_with_gn(module)


class ViT_Tiny(nn.Module):
    def __init__(self, num_classes=200):
        super(ViT_Tiny, self).__init__()
        # 使用torchvision提供的ViT模型，设置为16x16的patch，适配64x64图像
        self.model = models.vit_b_16(pretrained=False)  # 不使用预训练权重

        # 获取ViT的分类器部分的输入特征数
        in_features = self.model.heads[0].in_features  # 获取ViT的输入特征数

        # 修改头部（分类器）
        self.model.heads = nn.Sequential(  # 替换ViT的头部为自定义的全连接层
            nn.Linear(in_features, 512),  # 隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, num_classes)  # 输出层，适应Tiny-ImageNet 200个类别
        )

    def forward(self, x):
        return self.model(x)


class TuningModule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(TuningModule,self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels,kernel_size=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(out_channels,out_channels,kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=1)
    def forward(self,x):
        x = self.relu(self.conv(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class ModelWithTuning(nn.Module):
    def __init__(self,global_model,tuning_models):
        super(ModelWithTuning,self).__init__()
        self.global_model = global_model
        self.tuning_models = tuning_models
        self.fc = global_model.fc
    def forward(self,x,res_fs):
        features = {}
        for name,layer in self.global_model.named_children():
            if name == 'fc':
                continue
            x = layer(x)
            if name in self.tuning_models:
                input_f = torch.cat((x,res_fs[name].repeat(x.shape[0],1,1,1)),dim=1)
                x = self.tuning_models[name](input_f) + x
                features[name] = x
        x  = nn.functional.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x,1)
        x = self.fc(x)
        return features,x

class FusionBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class PromptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PromptBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

class FeatureFusionNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super(FeatureFusionNetwork,self).__init__()
        self.fusion_blocks = nn.ModuleList([FusionBlock(in_channels, out_channels) for in_channels, out_channels in zip(in_channels_list,out_channels_list)])
        self.prompt_blocks = nn.ModuleList([PromptBlock(out_channels, out_channels) for out_channels in out_channels_list])

    def forward(self, features1, features2):
        fused_features = []
        for i in range(len(features1)):
            # Concatenate corresponding feature maps from both inputs
            concatenated_features = torch.cat((features1[i], features2[i]), dim=1)
            # Apply fusion block
            fused_feature = self.fusion_blocks[i](concatenated_features)
            # Apply prompt block
            output_feature = self.prompt_blocks[i](fused_feature)
            fused_features.append(output_feature)
        return fused_features





class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)

        if self.name == 'mnist_2NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)

        if self.name == 'emnist_NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)

        if self.name == "mvtec_ad":
            self.n_cls = 15
            resnet18 = models.resnet18(pretrained=True)
            resnet18.fc = nn.Linear(resnet18.fc.in_features,self.n_cls)
            self.model = resnet18
        if self.name == 'sports':
            self.n_cls = 100
            efficientnet_b0 = models.efficientnet_b0()
            num_features = efficientnet_b0.classifier[1].in_features
            efficientnet_b0.classifier[1] = nn.Linear(num_features, 100)
            replace_bn_with_gn(efficientnet_b0)
            assert len(dict(efficientnet_b0.named_parameters()).keys()) == len(
                efficientnet_b0.state_dict().keys()), 'More BN layers are there...'

            self.model = efficientnet_b0

        if self.name == 'cifar10_LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'fer2013_LeNet':
            self.n_cls = 7
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 13 * 13, 560)
            self.fc2 = nn.Linear(560, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'cifar100_LeNet':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'cifar100_LeNet_ViT':
            self.n_cls = 200
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 13 * 13, self.n_cls)
            # self.fc2 = nn.Linear(384 * 2, 192 * 2)
            # self.fc3 = nn.Linear(192 * 2, self.n_cls)
        if self.name == 'celeba_four_class':
            self.n_labels = 2
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_labels)  # 输出两个标签

        def convert_relu_to_non_inplace(module):
            """
            Recursively convert all nn.ReLU(inplace=True) to nn.ReLU(inplace=False)
            to ensure Opacus compatibility.
            """
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU) and child.inplace:
                    setattr(module, name, nn.ReLU(inplace=False))
                else:
                    convert_relu_to_non_inplace(child)

        if self.name == 'Resnet18_7':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 7)
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            convert_relu_to_non_inplace(resnet18)  # <== 加在这里
            self.model = resnet18

        if self.name == 'Resnet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 10)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.conv1 = resnet18.conv1
            self.bn1 = resnet18.bn1
            self.relu = resnet18.relu
            self.maxpool = resnet18.maxpool
            self.layer1 = resnet18.layer1
            self.layer2 = resnet18.layer2
            self.layer3 = resnet18.layer3
            self.layer4 = resnet18.layer4
            self.avgpool = resnet18.avgpool
            self.fc = resnet18.fc

        if self.name == 'Resnet18_22':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 22)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18


        if self.name == 'Resnet18_100':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 100)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18

        if self.name == 'Resnet18_200':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 200)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18

        if self.name == 'Resnet18_mvtec':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 200)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18

        if self.name == 'ViT_Tiny':
            self.model = ViT(
                image_size = 64,
                patch_size= 16,
                num_classes= 200,
                dim = 128,
                depth = 10,
                heads = 8,
                mlp_dim = 256,
                dropout = 0.1,
                emb_dropout = 0.1,
            )
            # self.model = ViT_Tiny(num_classes=200)
            # self.model = ViT_Tiny(img_size=64, patch_size=16, num_classes=200)
            # ViT_Tiny = create_model('vit_tiny_patch16_224',pretrained=False,num_classes = 200)
            # self.model = ViT_Tiny

        if self.name == 'shakes_LSTM':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80

            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)

    def forward(self, x, is_feat=False):
        if self.name == 'Linear':
            x = self.fc(x)

        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            f1 = x
            x = F.relu(self.fc2(x))
            f2 = x
            x = self.fc3(x)
            if is_feat:
                return [f1,f2],x
            else:
                return x

        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            f1 = x
            x = F.relu(self.fc2(x))
            f2 = x
            x = self.fc3(x)
            if is_feat:
                return [f1,f2],x
            else:
                return x

        if self.name == 'mvtec_ad':
            x = self.model(x)

        if self.name == 'cifar10_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            f1 = x
            x = self.pool(F.relu(self.conv2(x)))
            f2 = x
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            f3 = x
            x = F.relu(self.fc2(x))
            f4 = x
            x = self.fc3(x)
            if is_feat:
                return [f1,f2,f3,f4],x
            else:
                return x
        if self.name == 'fer2013_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            f1 = x
            x = self.pool(F.relu(self.conv2(x)))
            f2 = x
            x = x.view(-1, 64 * 13 * 13)
            x = F.relu(self.fc1(x))
            f3 = x
            x = F.relu(self.fc2(x))
            f4 = x
            x = self.fc3(x)
            if is_feat:
                return [f1,f2,f3,f4],x
            else:
                return x
        if self.name == 'cifar100_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            f1 = x
            x = self.pool(F.relu(self.conv2(x)))
            f2 = x
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            f3 = x
            x = F.relu(self.fc2(x))
            f4 = x
            x = self.fc3(x)
            if is_feat:
                return [f1,f2,f3,f4],x
            else:
                return x

        if self.name == 'celeba_four_class':
            x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool
            x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool
            x = x.view(x.size(0), -1)  # flatten
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))  # 每个标签概率
            return x


        if self.name == 'cifar100_LeNet_ViT':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 13 * 13)
            x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = self.fc3(x)
        if self.name == 'Resnet18_7':
            x = self.model(x)
        if self.name == 'Resnet18':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            f1 = x
            x = self.layer2(x)
            f2 = x
            x = self.layer3(x)
            f3 = x
            x = self.layer4(x)
            f4 = x
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if is_feat:
                return {'layer1':f1, 'layer2':f2, 'layer3':f3, 'layer4':f4},x
            else:
                return x

        if self.name == 'Resnet18_100':
            x = self.model(x)

        if self.name == 'Resnet18_22':
            x = self.model(x)

        if self.name == 'Resnet18_200':
            x = self.model(x)
        if self.name == 'Resnet18_mvtec':
            x = self.model(x)
        if self.name == 'sports':
            x = self.model(x)

        if self.name == 'ViT_Tiny':
            x = self.model(x)

        if self.name == 'shakes_LSTM':
            x = self.embedding(x)
            x = x.permute(1, 0, 2)  # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1, :, :]
            x = self.fc(last_hidden)

        return x

# if __name__ == '__main__':
#     x = torch.randn(1,3,32,32)
#     model_func = lambda: client_model('Resnet18')
#     model1 = model_func()
#     feats, logits = model1(x,is_feat=True)
#     print(len(feats))
#     for i in range(len(feats)):
#         print(feats[i].shape)
#     print(logits.shape)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # feature1 = [torch.ones(50,64,16,16).to(device),torch.ones(50,64,8,8).to(device),torch.ones(50,128,4,4).to(device),torch.ones(50,256,2,2).to(device)]
    # feature2 = [torch.zeros(50,64,16,16).to(device),torch.zeros(50,64,8,8).to(device),torch.zeros(50,128,4,4).to(device),torch.ones(50,256,2,2).to(device)]
    # # for i in range(len(feature1)):
    # #     feature1[i].to(device)
    # #     feature2[i].to(device)
    # in_channels_list = [64,64,128,256]
    # out_channels_list = [64,64,128,256]
    # # list1 = [(in_channels, out_channels) for in_channels, out_channels in zip(in_channels_list,out_channels_list)]
    # # print('list1 is {}'.format(list1))
    # pr_model = FeatureFusionNetwork(in_channels_list=in_channels_list,out_channels_list=out_channels_list).to(device)
    # output = pr_model(feature1,feature2)
    # for i in range(len(output)):
    #     print(output[i].shape)


if __name__ == '__main__':
    tuning_modules = nn.ModuleDict({
        'layer1': TuningModule(64, 64),
        'layer2': TuningModule(128, 128),
        'layer3': TuningModule(256, 256),
        'layer4': TuningModule(512, 512)
    })
    global_model = models.resnet18(pretrained=True)
    features_map = {}


    def hook_fn(name):
        def hook(module,input,output):
            features_map[name] = output
        return hook
    global_model.layer1.register_forward_hook(hook_fn('layer1'))
    global_model.layer2.register_forward_hook(hook_fn('layer2'))
    global_model.layer3.register_forward_hook(hook_fn('layer3'))
    global_model.layer4.register_forward_hook(hook_fn('layer4'))




    for param in global_model.parameters():
        param.requires_grad = False

    model = ModelWithTuning(global_model,tuning_modules)

    tst_data = torch.randn(50,3,32,32)
    with torch.no_grad():
        _ = global_model(tst_data)
    for layer,feature_map in features_map.items():
        print(f'{layer}:{feature_map.shape}')
    features,logits = model(tst_data,features_map)
    print(logits.shape)
    for feature in features:
        print(f'{feature.shape}')
