import os
import torch
from torch.utils.data import Dataset
import torchvision

class PolyGen(Dataset):
    def __init__(self, root='/dataset/PolypGen/PolypGen2021_MultiCenterData_v3/', center = 1, transform=None, show_name=False):
        self.data = []
        self.label = []
        self.name = []
        if isinstance(center, int):
            center = [center]
        for c in center:
            files = os.listdir(os.path.join(root, 'data_C'+str(c), 'images_C'+str(c)))
            for f in files:
                self.data.append(os.path.join(os.path.join(root, 'data_C'+str(c), 'images_C'+str(c), f)))
                self.label.append(os.path.join(os.path.join(root, 'data_C'+str(c), 'masks_C'+str(c), f.replace('.jpg', '_mask.jpg'))))
                self.name.append(f)
        self.transform = transform
        self.show_name = show_name
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.data[idx])
        label = torchvision.io.read_image(self.label[idx], mode=torchvision.io.ImageReadMode.GRAY)
        if self.transform:
            image, label = torch.split(self.transform(torch.cat([image, label], dim=0)), [image.shape[0], label.shape[0]], dim=0)
        if self.show_name:
            return image, label, self.name[idx]
        else:
            return image, label
        
class Kvasir(Dataset):
    def __init__(self, root='/dataset/Kvasir-SEG/', transform=None, show_name=False):
        self.data = []
        self.label = []
        self.name = []
        files = os.listdir(os.path.join(root, 'images'))
        for f in files:
            self.data.append(os.path.join(os.path.join(root, 'images', f)))
            self.label.append(os.path.join(os.path.join(root, 'masks', f)))
            self.name.append(f)
        self.transform = transform
        self.show_name = show_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.data[idx])
        label = torchvision.io.read_image(self.label[idx], mode=torchvision.io.ImageReadMode.GRAY)
        if self.transform:
            image, label = torch.split(self.transform(torch.cat([image, label], dim=0)), [image.shape[0], label.shape[0]], dim=0)
        if self.show_name:
            return image, label, self.name[idx]
        else:
            return image, label