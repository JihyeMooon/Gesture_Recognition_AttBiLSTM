import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
import os
import os.path as osp

from opts import parse_opts_offline

opt = parse_opts_offline()

image_size = opt.image_size
num_frames = opt.num_frames

img_transform = Compose([
    Resize((image_size, image_size), Image.BILINEAR),
    ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class Gesturedata(Dataset): 
    def __init__(self, csv_file): 
        self.gesture_frame = pd.read_csv(csv_file, delimiter=' ')

    def __len__(self):
        return len(self.gesture_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        gesture_idx = self.gesture_frame.iloc[idx, 0]
        gesture_idx = int(gesture_idx[22:])
        label = int(self.gesture_frame.iloc[idx, 2])
        num_items = int(self.gesture_frame.iloc[idx, 1])
        dataset_dir = opt.video_path
        ges_dir = dataset_dir + str(gesture_idx)
        images = torch.zeros(1,3,num_items,image_size,image_size)
        for image_i in range(1, num_items+1):
            image_path = ges_dir + '/' + str(image_i).zfill(5) + '.jpg'
            img =  img_transform(Image.open(image_path))
            images[0,:,image_i-1,:,:]=img
        images = nn.functional.interpolate(images,[num_frames,image_size,image_size],mode='trilinear',align_corners=True)
        images = torch.squeeze(images)
        return  images, label
