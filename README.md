# 🔍 WS-SAM: Weakly-Supervised SAM for Object Detection and Segmentation

This repository contains a complete PyTorch implementation of the WS-SAM framework as described in the paper. 

- Adaptive Prompt Generator (ResNet + CAM + peak + offset)
- Bidirectional Adapter (inserted into SAM image encoder)
- Segmentation Mask Refinement (Hungarian + Cosine Similarity)
- Multiple Instance Detection Network (MIL classification)