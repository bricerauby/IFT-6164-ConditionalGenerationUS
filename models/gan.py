
from torch import cat, randn
import torch.nn as nn
from .resnet import ResNet18
class BasicBlock(nn.Module):
    def __init__(self, n_chans_in=32, n_chans_out=32, dim=3,act=nn.ReLU,
                 conv_kwargs={"kernel_size":3, "padding":1},  norm_kwargs={}, act_kwargs={}):
        super().__init__()
        if dim == 2 : 
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        elif dim==3: 
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        else :
            raise ValueError('incorrect dimension : {}'.format(dim))
        
        self.module_conv = nn.Sequential(conv(in_channels=n_chans_in,out_channels=n_chans_out,**conv_kwargs),
                                    norm(num_features=n_chans_out,
                                         **norm_kwargs),
                                    act(**act_kwargs),
                                    conv(in_channels=n_chans_out,
                                         out_channels=n_chans_out,
                                         **conv_kwargs),
                                    norm(num_features=n_chans_out,**norm_kwargs),
                                    act(**act_kwargs))
    def forward(self,x):
        x = self.module_conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, input_channel=2, n_chans_out=1, nblocs=3,dim=2) -> None:
        super().__init__() 
        self.upsample = nn.Upsample(mode='nearest', scale_factor=2)
        self.dim=dim
        self.down_blocks = []
        self.up_blocks = []
        n_chans_base = 256
        n_chans_in = input_channel
        for iBlock in range(nblocs):
            self.down_blocks.append(BasicBlock(n_chans_in=n_chans_in, n_chans_out=(2**iBlock)*n_chans_base, dim=dim))
            n_chans_in = (2**iBlock)*n_chans_base
            self.up_blocks.append(BasicBlock(n_chans_in= 2 * n_chans_in , n_chans_out=n_chans_in//2, dim=dim))
            
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.bottom_block = BasicBlock(n_chans_in=n_chans_in, n_chans_out=n_chans_in, dim=dim)
        if dim ==2:
            self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
            self.last_conv= nn.Conv2d(in_channels=n_chans_base//2,out_channels=n_chans_out ,kernel_size=1)
        elif dim==3:
            self.maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
            self.last_conv= nn.Conv3d(in_channels=n_chans_base//2,out_channels=n_chans_out ,kernel_size=1)
        else :
            raise ValueError('incorrect dimension : {}'.format(dim))
        self.final_act = nn.Sigmoid()
    def forward(self,x):
        skip_connections  = []
        for down_block in self.down_blocks:
            x = down_block(x)  
            skip_connections.append(x)
            x = self.maxpool(x)
        x =  self.bottom_block(x)
        for (upBlock,skip_connection) in zip(self.up_blocks[::-1], skip_connections[::-1]):
            x = self.upsample(x)
            x = cat((x, skip_connection), dim=-(self.dim+1))
            x = upBlock(x)
        x = self.last_conv(x)
        x = self.final_act(x)
        return x

class Generator(nn.Module):
    def __init__(self, conditionDim = 2, device='cuda', dim=2, unet_kwargs={}) -> None:
        super().__init__() 
        self.dim=dim
        self.unet = Unet(input_channel=(conditionDim+1), n_chans_out=2,dim=dim, **unet_kwargs)
        self.device=device
        self.default_size = [32,32]
        if dim ==3:
            self.default_size.append(16)
        self.to(device)
        print('n Parameters Unet', sum(p.numel() for p in self.unet.parameters() if p.requires_grad))
    def forward(self,z,x=None):
        if x is not None: 
            out = self.unet(cat((x,z),dim=-(self.dim+1)))
        else: 
            out = self.unet(z)
        return out

    def sample(self,shape,x=None):
        z = randn(*shape, device = self.device)
        if x is not None:
            return self.forward(z,x)
        else:
            return self.forward(z)
        
class Discriminator(nn.Module):
    def __init__(self, dim=2, in_chans=1) -> None:
        super().__init__()
        self.dim=dim
        if dim == 2 : 
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        elif dim==3: 
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        else :
            raise ValueError('incorrect dimension : {}'.format(dim))
        self.resnet = ResNet18(in_chans=in_chans, conv=conv, norm=norm)
        print('n Parameters resNet', sum(p.numel() for p in self.resnet.parameters() if p.requires_grad))
    def forward(self,x, cond =None):
        if cond is not None:
            return self.resnet(cat((x,cond),dim=-(self.dim+1)))
        else:
            return self.resnet(x)

