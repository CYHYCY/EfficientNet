# PyTorch implementation for EfficientNet
# class:
#       > Swish
#       > SEBlock
#       > MBConvBlock
#       > EfficientNet
# ---------------------------------------------------------------------------------------------------------------- #
# Author: Huijun Liu M.Sc.
# Date:   08.02.2020
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
from torch.nn import functional as F
from collections import OrderedDict

from torch import nn
import torch
import math


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Swish: Swish Activation Function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 groups=1, dilate=1):
        """
        conv + bn + swish
        in_planes: 输入的通道数
        out_planes: 输出的通道数
        kernel_size: 卷积核大小
        ...
        """
        super(ConvBlock, self).__init__()
        dilate = 1 if stride > 1 else dilate  # 膨胀卷积的参数，如果步长大于1，dilate为1
        padding = ((kernel_size - 1) // 2) * dilate

        self.conv_block = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilate, groups=groups, bias=False)),
            ("norm", nn.BatchNorm2d(num_features=out_planes,
                                    eps=1e-3, momentum=0.01)),
            ("act", Swish(inplace=True))
        ]))

    def forward(self, x):
        return self.conv_block(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([
            ("linear1", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
            ("act", Swish(inplace=True)),
            ("linear2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

    def forward(self, x):
        x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x, output_size=(1, 1))))
        return torch.mul(x, x_se)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MBConvBlock: MBConvBlock for EfficientNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MBConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 expand_ratio, kernel_size, stride, dilate,
                 reduction_ratio=4, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        # step 1. Expansion phase/Point-wise convolution
        if expand_ratio != 1:
            self.expansion = ConvBlock(in_planes, hidden_dim, 1)
            # 输出feature map高宽不变，通道数变为hidden_dim

        # step 2. Depth-wise convolution phase
        self.depth_wise = ConvBlock(hidden_dim, hidden_dim, kernel_size,
                                    stride=stride, groups=hidden_dim, dilate=dilate)
        # h=floor(h+1)/2,c=hidden_dim
        # step 3. Squeeze and Excitation
        if self.use_se:
            self.se_block = SEBlock(hidden_dim, reduced_dim)

        # step 4. Point-wise convolution phase
        self.point_wise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=hidden_dim,
                               out_channels=out_planes, kernel_size=1,
                               stride=1, padding=0, dilation=1, groups=1, bias=False)),
            ("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01))
        ]))
        # 高宽不变，通道数变为out_planes

    def forward(self, x):
        res = x

        # step 1. Expansion phase/Point-wise convolution
        if self.expand_ratio != 1:
            x = self.expansion(x)

        # step 2. Depth-wise convolution phase
        x = self.depth_wise(x)

        # step 3. Squeeze and Excitation
        if self.use_se:
            x = self.se_block(x)

        # step 4. Point-wise convolution phase
        x = self.point_wise(x)

        # step 5. Skip connection and drop connect
        if self.use_residual:  # 输入通道和输出通道相等，且stride为1时，做残差
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training, inplace=True)
            x = x + res

        return x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# EfficientNet: EfficientNet Implementation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class EfficientNet(nn.Module):
    def __init__(self, arch="b0", num_classes=1000):
        super(EfficientNet, self).__init__()

        arch_params = {
            # arch width_multi depth_multi input_h dropout_rate
            'b0': (1.0, 1.0, 224, 0.2),
            'b1': (1.0, 1.1, 240, 0.2),
            'b2': (1.1, 1.2, 260, 0.3),
            'b3': (1.2, 1.4, 300, 0.3),
            'b4': (1.4, 1.8, 380, 0.4),
            'b5': (1.6, 2.2, 456, 0.4),
            'b6': (1.8, 2.6, 528, 0.5),
            'b7': (2.0, 3.1, 600, 0.5),
        }
        width_multi, depth_multi, net_h, dropout_rate = arch_params[arch]

        settings = [
            # expand_ratio,channels,num_layers,kernel_size,stride,dilate
            # t, c, n, k, s, d
            [1, 16, 1, 3, 1, 1],  # 3x3, 112 -> 112    mod2的一些操作
            [6, 24, 2, 3, 2, 1],  # 3x3, 112 ->  56    mod3的一些操作
            [6, 40, 2, 5, 2, 1],  # 5x5, 56  ->  28    mod4的一些操作
            [6, 80, 3, 3, 2, 1],  # 3x3, 28  ->  14    mod5的一些操作
            [6, 112, 3, 5, 1, 1],  # 5x5, 14  ->  14   mod6的一些操作
            [6, 192, 4, 5, 2, 1],  # 5x5, 14  ->   7   mod7的一些操作
            [6, 320, 1, 3, 1, 1],  # 3x3, 7   ->   7   mod8的一些操作
        ]
        self.dropout_rate = dropout_rate
        out_channels = self._round_filters(32, width_multi)
        self.mod1 = ConvBlock(3, out_channels, kernel_size=3, stride=2, groups=1, dilate=1)
        # h=floor(h+1)/2,c=out_channels

        in_channels = out_channels
        drop_rate = self.dropout_rate
        mod_id = 0
        for t, c, n, k, s, d in settings:  # channel
            out_channels = self._round_filters(c, width_multi)
            repeats = self._round_repeats(n, depth_multi)

            if self.dropout_rate:
                drop_rate = self.dropout_rate * float(mod_id + 1) / len(settings)

            # Create blocks for module
            blocks = []
            for block_id in range(repeats):
                stride = s if block_id == 0 else 1  # 只有第一次时，步长为s，其余都是1
                dilate = d if stride == 1 else 1  # 只有第一次时，dilate为d，其余次数均为1

                blocks.append(("block%d" % (block_id + 1), MBConvBlock(in_channels, out_channels,
                                                                       expand_ratio=t, kernel_size=k,
                                                                       stride=stride, dilate=dilate,
                                                                       dropout_rate=drop_rate)))

                in_channels = out_channels
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        self.last_channels = self._round_filters(1280, width_multi)
        self.last_feat = ConvBlock(in_channels, self.last_channels, 1)

        self.classifier = nn.Linear(self.last_channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _make_divisible(value, divisor=8):
        """
        计算输出通道数
        """
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _round_filters(self, filters, width_multi):
        """
        计算输出的通道数
        """
        if width_multi == 1.0:
            return filters
        return int(self._make_divisible(filters * width_multi))

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        """
        计算重复次数
        """
        if depth_multi == 1.0:
            return repeats
        return int(math.ceil(depth_multi * repeats))

    def forward(self, x):
        x = self.mod2(self.mod1(x))  # 输出 (N, 16,   H/2,  W/2)
        x = self.mod3(x)  # (N, 24,   H/4,  W/4)
        x = self.mod4(x)  # (N, 32,   H/8,  W/8)
        x = self.mod6(self.mod5(x))  # (N, 96,   H/16, W/16)
        x = self.mod8(self.mod7(x))  # (N, 320,  H/32, W/32)
        x = self.last_feat(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.last_channels)
        if self.training and (self.dropout_rate is not None):
            x = F.dropout(input=x, p=self.dropout_rate,
                          training=self.training, inplace=True)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import os
    import time
    from torchstat import stat
    from pytorch_memlab import MemReporter

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    arch = "b3"
    img_preparam = {"b0": (224, 0.875),
                    "b1": (240, 0.882),
                    "b2": (260, 0.890),
                    "b3": (300, 0.904),
                    "b4": (380, 0.922),
                    "b5": (456, 0.934),
                    "b6": (528, 0.942),
                    "b7": (600, 0.949)}
    net_h = img_preparam[arch][0]
    model = EfficientNet(arch=arch, num_classes=1000)
    pretrain = True
    pretrain_dir = "./pretrain/"
    pretrain_weight_name = os.listdir(pretrain_dir)
    index = [item for item in pretrain_weight_name if arch in item][0]

    if pretrain:
        # state_dict1 = torch.load("./pretrain/efficientnet_b1_top1v_78.0.pkl")
        state_dict1 = torch.load(pretrain_dir + index)
        model.load_state_dict(state_dict1["model_state"])

    dummy_in = torch.randn(2, 3, net_h, net_h)
    dummy_out = model(dummy_in)
    print(dummy_in.shape)
    print(dummy_out.shape)

    # stat(model, (3, net_h, net_h))
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1,
    #                             momentum=0.90, weight_decay=1.0e-4, nesterov=True)
    # model = model.cuda().train()
    # loss_func = nn.CrossEntropyLoss().cuda()
    # dummy_in = torch.randn(2, 3, net_h, net_h).cuda().requires_grad_()
    # dummy_target = torch.ones(2).cuda().long().cuda()
    # reporter = MemReporter(model)
    #
    # optimizer.zero_grad()
    # dummy_out = model(dummy_in)
    # loss = loss_func(dummy_out, dummy_target)
    # print('========================================== before backward ===========================================')
    # reporter.report()
    #
    # loss.backward()
    # optimizer.step()
    # print('========================================== after backward =============================================')
    # reporter.report()
