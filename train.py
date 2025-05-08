import os
import yaml
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.ws_sam import WSSAM
from datasets.coco_loader import COCODataset
from datasets.voc_loader import VOCDataset
from utils.logger import Logger
from utils.metrics import iou
from utils.visualize import save_visualization

def get_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_dataloader(cfg):
    if cfg['data']['dataset'] == 'coco':
        dataset = COCODataset(cfg['data']['coco']['root'], cfg['data']['coco']['ann_file'])
    else:
        dataset = VOCDataset(cfg['data']['voc']['root'], cfg['data']['voc']['label_dir'], cfg['data']['voc']['class_names'])

    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset,
                        batch_size=cfg['train']['batch_size'],
                        num_workers=cfg['train']['num_workers'],
                        sampler=sampler)
    return loader

def main(args):
    cfg = get_config(args.config)
    dist.init_process_group(backend=cfg['device']['backend'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = WSSAM(num_classes=cfg['model']['num_classes'],
                  sam_type=cfg['model']['sam_type'],
                  sam_checkpoint=cfg['model']['sam_checkpoint']).to(device)
    model = DDP(model, device_ids=[local_rank])

    train_loader = build_dataloader(cfg)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg['train']['lr'],
                                momentum=cfg['train']['momentum'],
                                weight_decay=cfg['train']['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg['train']['scheduler']['milestones'],
                                                     gamma=cfg['train']['scheduler']['gamma'])

    logger = Logger(cfg['log']['log_dir']) if local_rank == 0 and cfg['log']['tensorboard'] else None

    best_iou = 0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        total_loss, total_iou = 0, 0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            boxes, scores, masks = model(images, labels)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iou += iou(masks > 0.5, masks > 0.5).mean().item()

            if local_rank == 0 and step % 10 == 0:
                print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")

        scheduler.step()

        if local_rank == 0:
            avg_loss = total_loss / len(train_loader)
            avg_iou = total_iou / len(train_loader)

            if logger:
                logger.log({'loss': avg_loss, 'iou': avg_iou}, epoch)

            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")

            # Save checkpoint
            os.makedirs(cfg['train']['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['train']['output_dir'], f"epoch_{epoch}.pth"))

            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(model.state_dict(), os.path.join(cfg['train']['output_dir'], "best_model.pth"))

    if logger:
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    main(args)
