#!/bin/bash

mkdir -p checkpoints

# Download SAM ViT-H checkpoint from Meta
echo "Downloading SAM ViT-H model..."
wget -O checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "Download completed. File saved to checkpoints/sam_vit_h_4b8939.pth"
