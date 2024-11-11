import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
from DDRNet_23_slim import get_seg_model
model = get_seg_model(cfg=None)

class CustomSegmentationDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_list = sorted(os.listdir(img_dir))
        self.label_list = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        label_path = os.path.join(self.label_dir, self.label_list[idx])

        image = Image.open(img_path).convert('RGB')  # Load image
        label = Image.open(label_path)  # Load segmentation mask

        if self.transform:
            image = self.transform(image)  # Apply transformations to image

        label = torch.from_numpy(np.array(label)).long()  # Convert label to tensor
        return image, label




train_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to a consistent size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Create dataset and dataloader
train_dataset = CustomSegmentationDataset(img_dir='/path/to/train/images',
                                          label_dir='/path/to/train/labels',
                                          # transform=train_transform
                                          )
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

