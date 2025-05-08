import torch
import torch.nn as nn
from models.prompt_generator import PromptGenerator
from models.adapter import BidirectionalAdapter
from models.refiner import SegmentationRefiner
from models.midn import MIDN
from models.sam_wrapper import load_sam_with_adapters
from torchvision.ops import masks_to_boxes

class WSSAM(nn.Module):
    def __init__(self, num_classes, sam_type='vit_h', sam_checkpoint=None):
        super(WSSAM, self).__init__()
        self.prompt_generator = PromptGenerator(num_classes)
        self.refiner = SegmentationRefiner()
        self.midn = MIDN(input_dim=256, num_classes=num_classes)
        self.sam = load_sam_with_adapters(model_type=sam_type, checkpoint=sam_checkpoint)

        for name, param in self.sam.image_encoder.named_parameters():
            if 'adapter' not in name and 'blocks.11' not in name:
                param.requires_grad = False

    def forward(self, image, labels=None):

        basic_prompts, auxiliart_prompts = self.prompt_generator(image)
        point_promts = torch.cat( basic_prompts, auxiliart_prompts)
        image_embedding = self.sam.image_encoder(image)
        prompt_embedding = self.sam.prompt_encoder(point_prompts)
        coarse_masks = self.sam.mask_decoder(image_embedding, prompt_embedding)

        with torch.no_grad():
            coarse_boxes = masks_to_boxes(coarse_masks > 0.5)
        box_prompt_embed = self.sam.prompt_encoder(coarse_boxes)
        refined_masks = self.sam.mask_decoder(image_embedding, box_prompt_embed)

        refined_masks = self.refiner.refine(refined_masks, prompt_embedding)
        refined_boxes = masks_to_boxes(refined_masks > 0.5)

        mask_feat = torch.mean(refined_masks, dim=(2, 3))
        scores = self.midn(mask_feat)

        return refined_boxes, scores, refined_masks
