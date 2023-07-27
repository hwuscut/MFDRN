import torch.nn as nn
from collections import OrderedDict
import torch


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)#######


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)#######
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),#####
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),#####
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25,kernel=3,):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, kernel)
        self.c2 = conv_layer(self.remaining_channels, in_channels, kernel)
        self.c3 = conv_layer(self.remaining_channels, in_channels, kernel)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, kernel)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused



class MRIDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25,):
        super(MRIDB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c2_1 = conv_layer(self.remaining_channels, in_channels, 5)
        self.c3_1 = conv_layer(self.remaining_channels, in_channels, 5)
        self.c4_1 = conv_layer(self.remaining_channels, in_channels, 5)
        self.c2_2 = conv_layer(self.remaining_channels, in_channels, 7)
        self.c3_2 = conv_layer(self.remaining_channels, in_channels, 7)
        self.c4_2 = conv_layer(self.remaining_channels, in_channels, 7)
        self.cout = conv_layer(in_channels*3,in_channels,1)
        # self.cout1 = conv_layer(48,16,1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.c6 = conv_block(304, in_channels, kernel_size=1, act_type='lrelu')
        # self.cca = CCALayer(self.distilled_channels * 4)
        #self.cca1 = CCALayer(self.distilled_channels)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c2 = self.act(self.c2(remaining_c1))
        out_c2_1 = self.act(self.c2_1(remaining_c1))
        out_c2_2 = self.act(self.c2_2(remaining_c1))
        #outcal_c2 = out_c2 + out_c2_1 + out_c2_2
        #outcal_c2 = self.cca(outcal_c2)
        #outcal_c2 = self.cca(self.cout(torch.cat([out_c2,out_c2_1,out_c2_2],dim=1)))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_1, remaining_c2_1 = torch.split(out_c2_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_2, remaining_c2_2 = torch.split(out_c2_2, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c3 = self.act(self.c3(remaining_c2))
        out_c3_1 = self.act(self.c3_1(remaining_c2_1))
        out_c3_2 = self.act(self.c3_2(remaining_c2_2))
        #outcal_c3 = out_c3 + out_c3_1 + out_c3_2
        #outcal_c3 = self.cca(outcal_c3)
        #outcal_c3 = self.cca(self.cout(torch.cat([out_c3,out_c3_1,out_c3_2],dim=1)))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_1, remaining_c3_1 = torch.split(out_c3_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_2, remaining_c3_2 = torch.split(out_c3_2, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c4 = self.c4(remaining_c3)
        out_c4_1 = self.c4_1(remaining_c3_1)
        out_c4_2 = self.c4_2(remaining_c3_2)
        #outcal_c4 = out_c4 + out_c4_1 + out_c4_2
        #outcal_c4 = self.cca(outcal_c4)
        #outcal_c4 = self.cca1(self.cout1(torch.cat([out_c4,out_c4_1,out_c4_2],dim=1)))

        out = self.c6(torch.cat([distilled_c1, distilled_c2, distilled_c2_1,distilled_c2_2,distilled_c3,distilled_c3_1,distilled_c3_2, out_c4,out_c4_1,out_c4_2], dim=1))
        out_fused = self.c5(self.cca(out)) + input
        # out_fused = self.c5(out) + input
        return out_fused


class IMDModuleadd2(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25,):
        super(IMDModuleadd2, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.c6 = conv_layer(self.distilled_channels*3,self.distilled_channels,3)
        self.c7 = conv_layer(self.distilled_channels*3,self.distilled_channels,5)
        self.c8 = conv_layer(self.distilled_channels*3,self.distilled_channels,7)
        self.c9 = conv_layer(self.distilled_channels*3,self.distilled_channels*4,3)
        self.c10 = conv_layer(48,16,1)

        self.cca = CCALayer(self.distilled_channels * 4)
    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c6(remaining_c1))
        out_c2_1 = self.act(self.c7(remaining_c1))
        out_c2_2 = self.act(self.c8(remaining_c1))
        remain_c2 =self.c9(torch.cat([out_c2,out_c2_1,out_c2_2],dim=1))
        distilled_c2, remaining_c2 = torch.split(remain_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c6(remaining_c2))
        out_c3_1 = self.act(self.c7(remaining_c2))
        out_c3_2 = self.act(self.c8(remaining_c2))
        remain_c3 =self.c9(torch.cat([out_c3,out_c3_1,out_c3_2],dim=1))
        #outcal_c3 = out_c3 + out_c3_1 + out_c3_2
        distilled_c3, remaining_c3 = torch.split(remain_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.act(self.c6(remaining_c3))
        out_c4_1 = self.act(self.c7(remaining_c3))
        out_c4_2 = self.act(self.c8(remaining_c3))
        outcal_c4 =self.c10(torch.cat([out_c4,out_c4_1,out_c4_2],dim=1))
        #outcal_c4 = out_c4 + out_c4_1 + out_c4_2
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, outcal_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        #out_fused = self.c5(out) + input
        return out_fused


class IMDModule8x8(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.125,):
        super(IMDModule8x8, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.c2_1 = conv_layer(self.remaining_channels, in_channels, 5)
        self.c3_1 = conv_layer(self.remaining_channels, in_channels, 5)
        self.c4_1 = conv_layer(self.remaining_channels, self.distilled_channels, 5)
        self.c2_2 = conv_layer(self.remaining_channels, in_channels, 7)
        self.c3_2 = conv_layer(self.remaining_channels, in_channels, 7)
        self.c4_2 = conv_layer(self.remaining_channels, self.distilled_channels, 7)
        # self.cout = conv_layer(in_channels*3,in_channels,1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.c6 = conv_layer(self.remaining_channels*3,self.remaining_channels,1)
        self.c7 = conv_layer(24,8,1)
        self.cca = CCALayer(self.distilled_channels * 8)
        #self.cca1 = CCALayer(self.distilled_channels)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c2 = self.act(self.c2(remaining_c1))
        out_c2_1 = self.act(self.c2_1(remaining_c1))
        out_c2_2 = self.act(self.c2_2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_1, remaining_c2_1 = torch.split(out_c2_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_2, remaining_c2_2 = torch.split(out_c2_2, (self.distilled_channels, self.remaining_channels), dim=1)
        remain_c2 = self.c6(torch.cat([remaining_c2,remaining_c2_1,remaining_c2_2],dim=1))

        out_c3 = self.act(self.c3(remain_c2))
        out_c3_1 = self.act(self.c3_1(remain_c2))
        out_c3_2 = self.act(self.c3_2(remain_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_1, remaining_c3_1 = torch.split(out_c3_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_2, remaining_c3_2 = torch.split(out_c3_2, (self.distilled_channels, self.remaining_channels), dim=1)
        remain_c3 = self.c6(torch.cat([remaining_c3,remaining_c3_1,remaining_c3_2],dim=1))

        out_c4 = self.c4(remain_c3)
        out_c4_1 = self.c4_1(remain_c3)
        out_c4_2 = self.c4_2(remain_c3)
        outcal_c4 = self.c7(torch.cat([out_c4,out_c4_1,out_c4_2],dim=1))


        out = torch.cat([distilled_c1, distilled_c2_1,distilled_c2_2,distilled_c2, distilled_c3, distilled_c3_1,distilled_c3_2,outcal_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        #out_fused = self.c5(out) + input
        return out_fused


class MFDRB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.125,):
        super(MFDRB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.c2_1 = conv_layer(self.remaining_channels, in_channels, 3,dilation=2)
        self.c3_1 = conv_layer(self.remaining_channels, in_channels, 3,dilation=2)
        self.c4_1 = conv_layer(self.remaining_channels, self.distilled_channels, 3,dilation=2)
        self.c2_2 = conv_layer(self.remaining_channels, in_channels, 3,dilation=3)
        self.c3_2 = conv_layer(self.remaining_channels, in_channels, 3,dilation=3)
        self.c4_2 = conv_layer(self.remaining_channels, self.distilled_channels, 3,dilation=3)
        # self.cout = conv_layer(in_channels*3,in_channels,1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, kernel_size=1)
        self.c6 = conv_layer(self.remaining_channels*3,self.remaining_channels,kernel_size=1)
        #self.c7 = conv_layer(24,8,1)
        self.c7 = conv_block(80, 64, kernel_size=1, act_type='lrelu')
        self.cca = CCALayer(self.distilled_channels * 8)
        #self.cca1 = CCALayer(self.distilled_channels)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c2 = self.act(self.c2(remaining_c1)) + remaining_c1##############
        out_c2_1 = self.act(self.c2_1(remaining_c1)) + remaining_c1
        out_c2_2 = self.act(self.c2_2(remaining_c1)) + remaining_c1
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_1, remaining_c2_1 = torch.split(out_c2_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_2, remaining_c2_2 = torch.split(out_c2_2, (self.distilled_channels, self.remaining_channels), dim=1)
        #remain_c2 = self.c6(torch.cat([remaining_c2,remaining_c2_1,remaining_c2_2],dim=1))

        out_c3 = self.act(self.c3(remaining_c2)) + remaining_c2
        out_c3_1 = self.act(self.c3_1(remaining_c2_1)) + remaining_c2
        out_c3_2 = self.act(self.c3_2(remaining_c2_2)) + remaining_c2
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_1, remaining_c3_1 = torch.split(out_c3_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_2, remaining_c3_2 = torch.split(out_c3_2, (self.distilled_channels, self.remaining_channels), dim=1)
        #remain_c3 = self.c6(torch.cat([remaining_c3,remaining_c3_1,remaining_c3_2],dim=1))

        out_c4 = self.c4(remaining_c3) + remaining_c3
        out_c4_1 = self.c4_1(remaining_c3_1) + remaining_c3
        out_c4_2 = self.c4_2(remaining_c3_2) + remaining_c3
        #outcal_c4 = self.c7(torch.cat([out_c4,out_c4_1,out_c4_2],dim=1))


        out = self.c7(torch.cat([distilled_c1, distilled_c2_1,distilled_c2_2,distilled_c2, distilled_c3, distilled_c3_1,distilled_c3_2,out_c4,out_c4_1,out_c4_2], dim=1))

        out_fused = self.c5(self.cca(out)) + input
        #out_fused = self.c5(out) + input
        return out_fused


class MRIDB_dilation(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.125,):
        super(MRIDB_dilation, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c2_1 = conv_layer(self.remaining_channels, in_channels,3,dilation=2)
        self.c3_1 = conv_layer(self.remaining_channels, in_channels,3,dilation=2)
        self.c4_1 = conv_layer(self.remaining_channels, in_channels,3,dilation=2)
        self.c2_2 = conv_layer(self.remaining_channels, in_channels,3,dilation=3)
        self.c3_2 = conv_layer(self.remaining_channels, in_channels,3,dilation=3)
        self.c4_2 = conv_layer(self.remaining_channels, in_channels,3,dilation=3)
        # self.cout = conv_layer(in_channels*3,in_channels,1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, kernel_size=1)
        self.c6 = conv_layer(self.remaining_channels*3,self.remaining_channels,kernel_size=1)
        #self.c7 = conv_layer(24,8,1)
        self.c7 = conv_block(248, 64, kernel_size=1, act_type='lrelu')
        self.cca = CCALayer(self.distilled_channels * 8)
        #self.cca1 = CCALayer(self.distilled_channels)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c2 = self.act(self.c2(remaining_c1))
        out_c2_1 = self.act(self.c2_1(remaining_c1))
        out_c2_2 = self.act(self.c2_2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_1, remaining_c2_1 = torch.split(out_c2_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c2_2, remaining_c2_2 = torch.split(out_c2_2, (self.distilled_channels, self.remaining_channels), dim=1)
        #remain_c2 = self.c6(torch.cat([remaining_c2,remaining_c2_1,remaining_c2_2],dim=1))

        out_c3 = self.act(self.c3(remaining_c2))
        out_c3_1 = self.act(self.c3_1(remaining_c2_1))
        out_c3_2 = self.act(self.c3_2(remaining_c2_2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_1, remaining_c3_1 = torch.split(out_c3_1, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c3_2, remaining_c3_2 = torch.split(out_c3_2, (self.distilled_channels, self.remaining_channels), dim=1)
        #remain_c3 = self.c6(torch.cat([remaining_c3,remaining_c3_1,remaining_c3_2],dim=1))

        out_c4 = self.c4(remaining_c3)
        out_c4_1 = self.c4_1(remaining_c3_1)
        out_c4_2 = self.c4_2(remaining_c3_2)
        #outcal_c4 = self.c7(torch.cat([out_c4,out_c4_1,out_c4_2],dim=1))


        out = self.c7(torch.cat([distilled_c1, distilled_c2_1,distilled_c2_2,distilled_c2, distilled_c3, distilled_c3_1,distilled_c3_2,out_c4,out_c4_1,out_c4_2], dim=1))

        out_fused = self.c5(self.cca(out)) + input
        #out_fused = self.c5(out) + input
        return out_fused


class IMDModule_speed(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class IMDModule_4(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.5,):
        super(IMDModule_4, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)

        self.c2 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN2 = conv_layer(in_channels, in_channels, 3)
        self.c3 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN3 = conv_layer(in_channels, in_channels, 3)
        self.c4 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN4 = conv_layer(in_channels, in_channels, 3)
        self.c5 = conv_layer(in_channels, self.remaining_channels, 3)

        self.c2_1 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN2_1 = conv_layer(in_channels, in_channels, 5)
        self.c3_1 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN3_1 = conv_layer(in_channels, in_channels, 5)
        self.c4_1 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN4_1 = conv_layer(in_channels, in_channels, 5)
        self.c5_1 = conv_layer(in_channels, self.remaining_channels, 3)

        self.c2_2 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN2_2 = conv_layer(in_channels, in_channels, 7)
        self.c3_2 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN3_2 = conv_layer(in_channels, in_channels, 7)
        self.c4_2 = conv_layer(in_channels, self.remaining_channels, 1)
        self.SRN4_2 = conv_layer(in_channels, in_channels, 7)
        self.c5_2 = conv_layer(in_channels, self.remaining_channels, 3)

        # self.cout = conv_layer(in_channels*3,in_channels,1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.act1 = activation('relu')
        # self.c5 = conv_layer(in_channels, in_channels, kernel_size=1)
        # self.c6 = conv_layer(self.remaining_channels*3,self.remaining_channels,kernel_size=1)
        #self.c7 = conv_layer(24,8,1)
        self.c7 = conv_block(384, 64, kernel_size=1, act_type='lrelu')
        # self.cca = CCALayer(64)
        #self.cca1 = CCALayer(self.distilled_channels)

    def forward(self, input):

        dout_c2 = self.act(self.c2(input))
        out_c2 = self.act1(self.SRN2(input) + input)

        dout_c2_1 = self.act(self.c2_1(input))
        out_c2_1 = self.act1(self.SRN2_1(input) + input)

        dout_c2_2 = self.act(self.c2_2(input))
        out_c2_2 = self.act1(self.SRN2_2(input) + input)


        dout_c3 = self.act(self.c3(out_c2))
        out_c3 = self.act1(self.SRN3(out_c2) + out_c2)

        dout_c3_1 = self.act(self.c3_1(out_c2_1))
        out_c3_1 = self.act1(self.SRN3_1(out_c2_1) + out_c2_1)

        dout_c3_2 = self.act(self.c3_2(out_c2_2))
        out_c3_2 = self.act1(self.SRN2_2(out_c2_2) + out_c2_2)


        dout_c4 = self.act(self.c4(out_c3))
        out_c4 = self.act1(self.SRN4(out_c3) + out_c3)
        out_c4_1_1 = self.act(self.c5(out_c4))

        dout_c4_1 = self.act(self.c4_1(out_c3_1))
        out_c4_1 = self.act1(self.SRN4_1(out_c3_1) + out_c3_1)
        out_c4_1_2 = self.act(self.c5(out_c4_1))

        dout_c4_2 = self.act(self.c4_2(out_c3_2))
        out_c4_2 = self.act1(self.SRN4_2(out_c3_2) + out_c3_2)
        out_c4_1_3 = self.act(self.c5(out_c4_2))



        out = self.c7(torch.cat([dout_c2, dout_c2_1, dout_c2_2, dout_c3, dout_c3_1, dout_c3_2, dout_c4, dout_c4_1, dout_c4_2, out_c4_1_1, out_c4_1_2, out_c4_1_3], dim=1))
        out_fused = out + input
        return out_fused

