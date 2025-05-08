import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self, img_dir, ann_path, transform=None):
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((1200, 800)),
            transforms.ToTensor()
        ])

        with open(ann_path, 'r') as f:
            coco = json.load(f)

        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = {cat['id']: cat['name'] for cat in coco['categories']}

        self.img_id_to_anns = {}
        for ann in self.annotations:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        anns = self.img_id_to_anns.get(img_info['id'], [])
        labels = set([ann['category_id'] for ann in anns])
        label_tensor = torch.zeros(len(self.categories))
        for l in labels:
            label_tensor[l - 1] = 1

        return image, label_tensor
