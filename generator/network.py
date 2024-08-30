import sys
sys.path.append(r'/home/diwah/programming/explainable-AI/classification/generator')
import gc

sys.path.append(r'../')
from generator.unet import *
from generator.vcc import VariableChannelConvolution
import generator.feature_extractor as feature_extractor

class ParallelUNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(ParallelUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        #input image
        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        #feature maps
        self.inc_f = (DoubleConv(n_channels, 64))
        self.down1_f = (Down(64, 128))
        self.down2_f = (Down(128, 256))
        self.down3_f = (Down(256, 512))
        self.down4_f = (Down(512, 1024 // factor))
        self.up1_f = (Up(1024, 512 // factor, bilinear))
        self.up2_f = (Up(512, 256 // factor, bilinear))
        self.up3_f = (Up(256, 128 // factor, bilinear))
        self.up4_f = (Up(128, 64, bilinear))
    def forward(self, x,f):
        f1 = self.inc_f(f)
        x1 = self.inc(x) +f1

        f2 = self.down1_f(f1)
        x2 = self.down1(x1)+f2

        f3 = self.down2_f(f2)
        x3 = self.down2(x2) +f3

        f4 = self.down3_f(f3)
        x4 = self.down3(x3) +f4

        f5 = self.down4_f(f4)
        x5 = self.down4(x4)+f5

        f_up = self.up1_f(f5,f4)
        x = self.up1(x5, x4) +f_up

        f_up = self.up2_f(f_up,f3)
        x = self.up2(x, x3)+f_up

        f_up = self.up3_f(f_up,f2)
        x = self.up3(x, x2)+f_up

        f_up = self.up4_f(f_up,f1)
        x = self.up4(x, x1)+f_up

        logits = self.outc(x)
        return logits

class ExplanationNetwork(nn.Module):
    def __init__(self,n_channels,fm_kernels=3, fm_padding = 1, bilinear=False):
        super(ExplanationNetwork, self).__init__()
        self.generator = ParallelUNet(n_channels,1,bilinear)
        # self.fmp_unitwise = VariableChannelConvolution(1,1)
        self.fmp_layerwise = VariableChannelConvolution(1,n_channels)
        
    def forward(self,inp,f):
        # f = self.extractor(inp)
        # stacked_features = None  # Initialize the stacked features tensor
        # for i in f.values():
        #     i = utils.utility.resize_array(i,(64,64))
        #     features = self.fmp_unitwise(i)
        #     del i
        #     gc.collect()
        #     if stacked_features is None:
        #         stacked_features = features
        #     else:
        #         stacked_features = torch.cat((stacked_features, features), dim=0)
            
        # del f
        # gc.collect()
        output = self.fmp_layerwise(f)
        output = self.generator(inp,output)
        return output