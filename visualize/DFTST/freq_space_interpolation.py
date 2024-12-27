# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:25:46 2024

@author: pky0507
"""
import numpy as np
import torch
from threshold import soft_threshold

def extract_amp_spectrum(trg_img):
    return torch.abs(torch.fft.fft2(trg_img))

def amp_spectrum_swap(amp_local, amp_target, L=0.1 , ratio=0, threshold_ratio=0.05):
    if isinstance(ratio, list):
        ratio = np.random.uniform(ratio[0], ratio[1])
    a_local = torch.fft.fftshift( amp_local)
    a_trg = torch.fft.fftshift( amp_target)
    h = a_local.shape[-2]
    w = a_local.shape[-1]
    b = (np.floor(np.amin((h,w))*L)).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1
    threshold = torch.amax(a_trg, dim=[-1, -2], keepdim=True)*threshold_ratio
    a_local[...,h1:h2,w1:w2] = a_local[...,h1:h2,w1:w2] * ratio + soft_threshold(a_trg[...,h1:h2,w1:w2], threshold) * (1- ratio)
    a_local = torch.fft.ifftshift( a_local)
    return a_local

def freq_space_interpolation(local_img, amp_target, L=0.1 , ratio=0):
    # get fft of local sample
    fft_local = torch.fft.fft2(local_img)

    # extract amplitude and phase of local sample
    amp_local, pha_local = torch.abs(fft_local), torch.angle(fft_local)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * torch.exp( 1j * pha_local )
    local_in_trg = torch.fft.ifft2(fft_local_)
    local_in_trg = torch.real(local_in_trg)

    return local_in_trg

def freq_space_interpolation_batch(local_img, target_set:list, L=0.1, ratio=[0, 1]):
    n_image = local_img.shape[0]
    n_target = len(target_set)
    target_images = []
    for i in range(n_image):
        target = target_set[np.random.randint(n_target)]
        target_images.append(target.__getitem__(np.random.randint(len(target)))[0])
    amp_target = extract_amp_spectrum(torch.stack(target_images))
    local_in_trg = freq_space_interpolation(local_img, amp_target, L, ratio)
    return local_in_trg