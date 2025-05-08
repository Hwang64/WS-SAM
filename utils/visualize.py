import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    image = TF.to_pil_image(image.cpu())
    mask = mask.cpu().numpy()
    overlay = Image.new('RGBA', image.size, color + (0,))
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] > 0.5:
                overlay.putpixel((x, y), color + (int(alpha * 255),))
    return Image.alpha_composite(image.convert('RGBA'), overlay)

def draw_boxes(image, boxes, labels=None):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        draw.rectangle([x1, y1, x2, y2], outline='yellow', width=2)
        if labels:
            draw.text((x1, y1), str(labels[i]), fill='white')
    return image

def save_visualization(image, mask, boxes, labels, save_path):
    image = overlay_mask(image, mask)
    image = draw_boxes(image, boxes, labels)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
