import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import csv
from PIL import Image

class SampleDataset(Dataset):
    def __init__(self, image_dir, conductivities, wts, transform=None):
        self.conds=conductivities
        self.sp_vols=[]
        self.wts=wts
        self.samples=[]
        for filename in sorted(os.listdir(image_dir), key=lambda x: int(x[:-5])):
            img=transform(Image.open(image_dir+filename))
            self.samples.append(img)
            self.sp_vols.append(torch.sum(img)/(img.shape[1]*img.shape[2]))
        self.samples=torch.stack(tensors=self.samples)
        self.sp_vols=torch.tensor(self.sp_vols, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index], self.sp_vols[index], self.wts[index], self.conds[index]
    
    def getWts(self):
        return self.wts
    
    def to(self, device):
        self.samples=self.samples.to(device)
        self.conds=self.conds.to(device)
        self.wts=self.wts.to(device)
        self.sp_vols=self.sp_vols.to(device)

def load_dataset(csvFilePath, imageDirectoryPath):
    transform=transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
    ])
    with open(csvFilePath) as f:
        reader=csv.reader(f)
        d=list(reader)
    conductivities=torch.tensor(data=[float(x[1]) for x in d[:]], dtype=torch.float32)
    wts=torch.tensor(data=[float(x[0]) for x in d[:]], dtype=torch.float32)
    dataset=SampleDataset(imageDirectoryPath, conductivities, wts, transform)
    dataset.to('cuda')
    return dataset