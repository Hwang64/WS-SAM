import os
import torch
import argparse
import yaml
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.ws_sam import WSSAM
from datasets.coco_loader import COCODataset
from datasets.voc_loader import VOCDataset
from utils.visualize import save_visualization

@torch.no_grad()
def inference(cfg, model, dataloader, device):
    model.eval()
    os.makedirs("outputs/vis", exist_ok=True)

    for idx, (image, labels) in enumerate(dataloader):
        image = image.to(device)
        labels = labels.to(device)

        boxes, scores, masks = model(image)

        # 一张图为例（可扩展为多图）
        for i in range(image.size(0)):
            img = image[i].cpu()
            box = boxes[i].cpu().unsqueeze(0)
            mask = masks[i].squeeze().cpu()
            label = torch.argmax(scores[i]).item()

            save_path = f"outputs/vis/image_{idx:03d}.png"
            save_visualization(img, mask, box, [label], save_path)
            print(f"Saved: {save_path}")

def get_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_dataset(cfg):
    if cfg['data']['dataset'] == 'coco':
        return COCODataset(cfg['data']['coco']['root'], cfg['data']['coco']['ann_file'])
    else:
        return VOCDataset(cfg['data']['voc']['root'],
                          cfg['data']['voc']['label_dir'],
                          cfg['data']['voc']['class_names'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()

    cfg = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = WSSAM(num_classes=cfg['model']['num_classes'],
                  sam_type=cfg['model']['sam_type'],
                  sam_checkpoint=cfg['model']['sam_checkpoint']).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    inference(cfg, model, dataloader, device)
