import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(512, 256, kernel_size,padding=padding)
        self.conv3 = nn.Conv2d(256, 128, kernel_size,padding=padding)
        self.conv4 = nn.Conv2d(128, 64, kernel_size,padding=padding)
        self.conv5 = nn.Conv2d(64, out_channels, kernel_size,padding=padding)
        self.gelu = nn.GELU()
    def forward(self,x):
    
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = self.gelu(self.conv4(x))
        x = self.gelu(self.conv5(x))
        

        return x

class VariableChannelConvolution(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=3,padding=1):
        super().__init__()
        self.conv_block = ConvBlock(in_channels,out_channels,kernel_size,padding=padding)

    def forward(self, x):
        num_maps = x.shape[1]
        running_sum = 0
        for n in range(num_maps):
            running_sum += self.conv_block(torch.unsqueeze(x[:,n,:,:],axis=1))
        averaged_feature = running_sum / num_maps
        return averaged_feature