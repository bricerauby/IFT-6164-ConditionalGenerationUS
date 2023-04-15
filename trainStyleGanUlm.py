import sys
sys.path.append("stylegan3")
from stylegan3.training.networks_stylegan2 import Generator
from stylegan3.training.networks_stylegan2 import Discriminator


gen = Generator(z_dim=32,c_dim=1,w_dim=128,img_resolution=32,img_channels=1)
disc = Discriminator(c_dim=1,img_resolution=32,img_channels=1)



