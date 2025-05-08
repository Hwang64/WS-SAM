
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from models.adapter import BidirectionalAdapter

def insert_adapters_into_image_encoder(image_encoder, proj_dim=256):
    for name, module in image_encoder.named_modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            in_dim = module.linear1.in_features
            adapter = BidirectionalAdapter(in_dim=in_dim, proj_dim=proj_dim)
            module.adapter = adapter  # insert adapter as a new submodule

            original_forward = module.forward

            def modified_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
                src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask, need_weights=False)[0]
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                mha_out = src
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)
                src = self.adapter(src, mha_out)
                return src

            # Patch the forward method
            module.forward = modified_forward.__get__(module)

def load_sam_with_adapters(model_type='vit_h', checkpoint='checkpoints/sam_vit_h_4b8939.pth', adapter_dim=256):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    insert_adapters_into_image_encoder(sam.image_encoder, proj_dim=adapter_dim)
    return sam
