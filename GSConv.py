import torch
import torch.nn as nn

def shuffle_channel(x, num_groups):
    """channel shuffle 的常规实现
    """
    batch_size, num_channels, height, width = x.size()
    assert num_channels % num_groups == 0

    x = x.view(batch_size, num_groups, num_channels // num_groups, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    return x.contiguous().view(batch_size, num_channels, height, width)

class Conv(nn.Module):   # 默认不进行下采样的普通3*3卷积
    def __init__(self,in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, act = True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Sequential()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module): # 深度可分离卷积 默认不下采样
    def __init__(self,in_channels, out_channels, kernel_size = 3,stride = 1, padding = 1, act = True):
        super(DWConv,self).__init__()
        self.dwconv = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,
                                stride = stride, padding=padding, groups=in_channels)
        self.pwconv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,
                                stride=1,padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self,x):
        x = self.dwconv(x)
        # print("in DWConv", x.shape)
        x = self.pwconv(x)
        # print("in DWConv", x.shape)
        x = self.bn(x)
        # print("in DWConv", x.shape)
        x = self.act(x)
        # print("in DWConv", x.shape)
        return x

class GSConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,act = True):
        super(GSConv, self).__init__()
        self.conv = Conv(in_channels, out_channels//2, kernel_size, stride=stride, padding=padding ,act = act)
        self.dwconv = DWConv(out_channels//2, out_channels//2, kernel_size, stride=1, padding=padding ,act=act)

    def forward(self,x):
        x = self.conv(x)
        # print("in GSConv",x.shape)
        x1 = self.dwconv(x)
        # print("in GSConv", x1.shape)
        x = torch.cat((x,x1),dim=1)
        _,C,_,_ = x.shape
        x = shuffle_channel(x,C//2)
        # print("in GSConv", x.shape)
        return x


class GSBottleNeck(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size = 3,stride = 1, padding = 1,act = True):
        super(GSBottleNeck, self).__init__()
        self.gsconv1 = GSConv(in_channels,out_channels,kernel_size,stride=1,padding=padding,act = act)
        self.gsconv2 = GSConv(out_channels,out_channels,kernel_size = kernel_size, stride=stride,padding=padding,act=act)

        # shortcut
        if (in_channels == out_channels and stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                       padding=kernel_size//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self,x):
        x_res = self.gsconv1(x)
        # print("in GSBottleNeck",x_res.shape)
        x_res = self.gsconv2(x_res)
        # print("in GSBottleNeck", x_res.shape)
        x = self.shortcut(x)
        # print("in GSBottleNeck", x.shape)
        return x+x_res          # shortcut

class VoVGSConv(nn.Module):
    # Slim-neck by GSConv
    def __init__(self,in_channels,out_channels,kernel_size = 3,stride = 1, padding = 1,act = True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(VoVGSConv, self).__init__()
        self.conv1 = Conv(in_channels, in_channels//2, kernel_size=kernel_size, stride=stride, padding = padding,act = act)
        self.GSconv1 = GSConv(in_channels//2, in_channels//2, kernel_size=kernel_size, stride=1, padding = padding,act = act)
        self.GSconv2 = GSConv(in_channels//2, in_channels//2, kernel_size=kernel_size, stride=1, padding = padding,act = act)
        self.conv2 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding = padding,act = act)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.GSconv1(x)
        x1 =  self.GSconv2(x1)
        x = torch.cat((x,x1), 1)
        x = self.conv2(x)
        return x

if __name__ == "__main__":

    C1 = 16
    C2 = 32
    conv_layer = GSBottleNeck(C1,C2,stride=1)
    # print(conv_layer)
    x = torch.randn(1,16,224,224)
    y = conv_layer(x)

    print("###################################")

    conv_layer2 = VoVGSConv(C1, C2, stride=2)
    y2 = conv_layer2(x)
    print("final output",y.shape,y2.shape)
