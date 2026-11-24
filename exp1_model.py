import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class Exp1Model(nn.Module):
    def __init__(self, shared_dim=512, device="cuda"):
        super().__init__()
        self.device = device
        self.image_encoder = timm.create_model("fastvit_mci2", pretrained=False, num_classes=0)
        img_dim = 1280                          
        txt_dim = 768                           

        self.image_proj = Projector(img_dim, shared_dim)   
        self.text_proj  = Projector(txt_dim, shared_dim)   

    def image_transform(self, image_size=256, training=True):
        if training:
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandAugment(),
                transforms.ToTensor(),          
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def encode_image(self, images):
        feats = self.image_encoder(images)      
        feats = self.image_proj(feats)          
        return F.normalize(feats, dim=-1)

    def encode_text(self, text_feats):
        feats = self.text_proj(text_feats)     
        return F.normalize(feats, dim=-1)
    
