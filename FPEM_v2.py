import torch.nn as nn
import torch.nn.functional as F
import math
import torch

# import模块时只导入__all__中的内容
__all__ = ["FPEM_v2"]
class Conv_BN_ReLU(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        self.out_channels = out_channels
        planes = out_channels
        # reduce layers
        self.in1_conv = nn.Conv2d(in_channels[0], planes, kernel_size=1, bias=False, groups=in_channels[0])
        self.in2_conv = nn.Conv2d(in_channels[1], planes, kernel_size=1, bias=False, groups=in_channels[1])
        self.in3_conv = nn.Conv2d(in_channels[2], planes, kernel_size=1, bias=False, groups=in_channels[2])
        # self.in4_conv = nn.Conv2d(in_channels[3], planes, kernel_size=1, bias=False, groups=in_channels[3])
        self.in4_conv = nn.Conv2d(in_channels[3], planes, kernel_size=1, bias=False)
        # smooth layers
        # print(planes, planes//4)
        self.f4_conv = nn.Conv2d(planes, planes // 4, kernel_size=3, padding=1, bias=False)
        self.f3_conv = nn.Conv2d(planes, planes // 4, kernel_size=3, padding=1, bias=False)
        self.f2_conv = nn.Conv2d(planes, planes // 4, kernel_size=3, padding=1, bias=False)
        self.f1_conv = nn.Conv2d(planes, planes // 4, kernel_size=3, padding=1, bias=False)


        self.dwconv3_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)  #深度可分离卷积
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4, p5):
        p3 = F.interpolate(p3, scale_factor=2)
        p4 = F.interpolate(p4, scale_factor=4)
        p5 = F.interpolate(p5, scale_factor=8)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def forward(self, x):
        f1, f2, f3, f4 = x
        f1 = self.in1_conv(f1)
        f2 = self.in2_conv(f2)
        f3 = self.in3_conv(f3)
        f4 = self.in4_conv(f4)

        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        # print("f3_: ",f3_.shape)
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        # print("f2_: ",f2_.shape)
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))
        # print("f1_: ",f1_.shape)
        f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_,f1_)))
        # print("f2_: ", f2_.shape)
        f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_,f2_)))
        # print("f3_: ", f3_.shape)
        f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))
        # print("f4_: ", f4_.shape)

        f1 = f1 + f1_
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4 + f4_

        f1 = self.f1_conv(f1)
        f2 = self.f2_conv(f2)
        f3 = self.f3_conv(f3)
        f4 = self.f4_conv(f4)

        x = self._upsample_cat(f1,f2,f3,f4)
        return x

if __name__ == "__main__":

    import torch
    import math
    from torchsummaryX import summary
    f4 = torch.randn(2,256,20,20)
    f3 = torch.randn(2,128,40,40)
    f2 = torch.randn(2,64,80,80)
    f1 = torch.randn(2,32,160,160)
    x = []
    x.append(f1)
    x.append(f2)
    x.append(f3)
    x.append(f4)
    ffpm = FPEM_v2(in_channels=[32,64,128,256],out_channels=256)
    summary(ffpm, x)  # 输出网络结构
    y = ffpm(x)
    print(y.shape)
