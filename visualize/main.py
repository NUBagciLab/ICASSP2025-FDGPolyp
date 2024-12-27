# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:02:25 2024

@author: pky0507
"""

import torch
from torchvision.transforms import v2
from data_loader import PolyGen
from torchvision.utils import save_image
from freq_space_interpolation_ELCFS import extract_amp_spectrum
from freq_space_interpolation_ELCFS import freq_space_interpolation as freq_space_interpolation_ELCFS
from freq_space_interpolation_DFTST import freq_space_interpolation as freq_space_interpolation_DFTST
from freq_space_interpolation_DFTHT import freq_space_interpolation as freq_space_interpolation_DFTHT

transform =   v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.ToDtype(torch.float32, scale=True)]) 
dataset1 = PolyGen(root='/dataset/PolypGen/PolypGen2021_MultiCenterData_v3/', center = 1, transform = transform)
dataset2 = PolyGen(root='/dataset/PolypGen/PolypGen2021_MultiCenterData_v3/', center = 3, transform = transform)
img1 = dataset1.__getitem__(0)[0]
img2 = dataset2.__getitem__(2)[0]
img_ELCFS = freq_space_interpolation_ELCFS(img1, extract_amp_spectrum(img2), ratio=0.5)
img_DFTST = freq_space_interpolation_DFTST(img1, extract_amp_spectrum(img2), ratio=0.5)
img_DFTHT = freq_space_interpolation_DFTHT(img1, extract_amp_spectrum(img2), ratio=0.5)
save_image(img1, 'source.jpg')
save_image(img2, 'target.jpg')
save_image(img_ELCFS, 'elcfs.jpg')
save_image(img_DFTST, 'dftst.jpg')
save_image(img_DFTHT, 'dftht.jpg')

img1_dft = torch.fft.fftshift(torch.fft.fft2(img1))
img1_amp = torch.abs(img1_dft)
img1_angle = torch.angle(img1_dft)
save_image(img1_amp/torch.max(img1_amp)*255, 'source_dft.jpg')
save_image(img1_angle, 'source_dft_angle.jpg')
img2_dft = torch.fft.fftshift(torch.fft.fft2(img2))
img2_amp = torch.abs(img2_dft)
img2_angle = torch.angle(img2_dft)
save_image(img2_amp/torch.max(img2_amp)*255, 'target_dft.jpg')
save_image(img2_angle, 'target_dft_angle.jpg')
img_DFTHT_dft = torch.fft.fftshift(torch.fft.fft2(img_DFTHT))
img_DFTHT_amp = torch.abs(img_DFTHT_dft)
save_image(img_DFTHT_amp/torch.max(img_DFTHT_amp)*255, 'dftht_dft.jpg')