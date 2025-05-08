# === WS-SAM CONFIGURATION FILE ===

model:
  name: ws_sam
  sam_type: vit_h
  sam_checkpoint: checkpoints/sam_vit_h_4b8939.pth
  num_classes: 20  # VOC: 20, COCO: 80 → 可在运行时覆盖
  use_self_prompting: true

train:
  batch_size: 8
  num_workers: 4
  optimizer: sgd
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  scheduler:
    type: multistep
    milestones: [8, 11]
    gamma: 0.1
  epochs: 12
  eval_interval: 1
  save_interval: 1
  resume: false
  output_dir: checkpoints

log:
  tensorboard: true
  log_dir: logs/ws_sam

data:
  dataset: coco  # or voc
  coco:
    root: /home/server01/DatasSet/COCO
    ann_file: /home/server01/DatasSet/COCO/annotations/instances_train2017.json
  voc:
    root: /home/server01/DataSet/VOC
    label_dir: /home/server01/DataSet/VOC/labels
    class_names: ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

device:
  use_ddp: true
  gpu_ids: [0, 1, 2, 3]
  backend: nccl
