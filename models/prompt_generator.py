import torch
import torch.nn as nn
import torchvision
from torchvision.ops import DeformConv2d

class ResNetCAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        feature_map = self.features(x)  # [B, 2048, 7, 7]
        pooled = self.avgpool(feature_map)  # [B, 2048, 1, 1]
        pooled = torch.flatten(pooled, 1)
        logits = self.fc(pooled)
        
        # Generate CAM using predicted class weights
        weights = self.fc.weight  # [num_classes, 2048]
        _, preds = torch.max(logits, dim=1)
        selected_weights = weights[preds]  # [B, 2048]
        
        # Compute CAM
        cam = torch.einsum('bc,bchw->bhw', selected_weights, feature_map)
        cam = nn.functional.relu(cam)
        return cam, logits

class TONNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 
                            kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(in_channels, 18,  # 9 points * 2 offsets
                                    kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        offsets = self.offset_conv(x)
        return offsets

class PromptGenerator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cam_generator = ResNetCAM(num_classes)
        self.ton_network = TONNetwork()
        
    def get_max_coords(self, cam):
        batch_size, H, W = cam.shape
        cam_flat = cam.view(batch_size, -1)
        idx = torch.argmax(cam_flat, dim=1)
        h = (idx // W).float()
        w = (idx % W).float()
        return torch.stack([w, h], dim=1)  # Return as [x, y] coordinates
        
    def forward(self, x):
        cam, logits = self.cam_generator(x)
        max_coords = self.get_max_coords(cam)
        
        # Process CAM through TON network
        cam = cam.unsqueeze(1)  # Add channel dimension
        offsets = self.ton_network(cam)
        
        return max_coords, offsets
