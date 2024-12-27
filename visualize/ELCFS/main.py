# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:02:25 2024

@author: pky0507
"""

import torch
from torchvision.transforms import v2
from data_loader import PolyGen
from torchvision.utils import save_image
from freq_space_interpolation import freq_space_interpolation, extract_amp_spectrum

transform =   v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.ToDtype(torch.float32, scale=True)]) 
dataset1 = PolyGen(root='/dataset/PolypGen/PolypGen2021_MultiCenterData_v3/', center = 1, transform = transform)
dataset2 = PolyGen(root='/dataset/PolypGen/PolypGen2021_MultiCenterData_v3/', center = 3, transform = transform)
img1 = dataset1.__getitem__(0)[0]
img2 = dataset2.__getitem__(2)[0]
img3 = freq_space_interpolation(img1, extract_amp_spectrum(img2), ratio=0.5)
save_image(img1, 'source.jpg')
save_image(img2, 'target.jpg')
save_image(img3, 'result.jpg')
