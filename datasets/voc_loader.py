import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VOCDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.class_names = class_names
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((1200, 800)),
            transforms.ToTensor()
        ])

        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        image = self.transform(image)

        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        label_tensor = torch.zeros(len(self.class_names))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                classes = f.read().strip().split()
                for c in classes:
                    if c in self.class_names:
                        label_tensor[self.class_names.index(c)] = 1

        return image, label_tensor
