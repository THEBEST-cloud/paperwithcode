#特征金字塔：FPN
'''
输入四层特征图 输出四层特征图
'''

import  numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class FPN(nn.Module):
    def __init__(self):
        super(FPN,self).__init__()
        self.inplanes = 64

        #对C5减少通道数得到P5
        self.toplayer = nn.Sequential(nn.Conv2d(512,256,1,1,0),
                                        nn.Conv2d(256,256,1,1,0))

        #3x3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)

        #横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(256, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(128, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(64, 256, 1, 1, 0)

    #自上而下的上采样模块
    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.interpolate(x,size=(H,W),mode='bilinear',align_corners=False) + y
    def forward(self,x):
        # 自下而上的输入 x为特征列表
        c2,c3,c4,c5 = x

        print(c2.shape,c3.shape,c4.shape,c5.shape)

        #自上而下
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3)) 
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        #卷积融合，平滑处理
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5

if __name__ == "__main__":
    fpn = FPN()

    c2 = torch.randn(1,64,512,512)
    c3 = torch.randn(1,128,256,256)
    c4 = torch.randn(1,256,128,128)
    c5 = torch.randn(1,512,64,64)

    x = [c2,c3,c4,c5]

    y = fpn(x)

    print(y[0].shape,y[1].shape,y[2].shape,y[3].shape)

