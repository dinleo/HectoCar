import torch
import torch.nn as nn
from models.model_utils import load_model, safe_init

class Hecto(nn.Module):
    def __init__(self, detr_backbone, device="cuda", num_classes=396):
        super().__init__()
        self.device = device
        self.threshold = 0.3
        self.detr_backbone = detr_backbone

        # Linear projection for image features to match D
        self.img_proj = nn.ModuleList([
            nn.Linear(c, 256) for c in [256, 256, 256, 256]
        ])
        self.img_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(4)
        ])

        # Self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.freeze_backbone()

    def forward(self, batched_input):
        with torch.no_grad():
            self.detr_backbone.eval()
            detr_output = self.detr_backbone(batched_input)

        pred_logits = detr_output["pred_logits"].sigmoid()  # (B, Q, 128)
        hs = detr_output["hs"]                               # (B, Q, D)
        image_features = detr_output["image_feature"]        # list of 4 x [B, C_i, H, W]

        B, Q, D = hs.shape
        logits_list = []

        for b in range(B):
            fg_mask = (pred_logits[b] > self.threshold).any(dim=-1)  # (Q,)
            selected_hs = hs[b][fg_mask]  # (M, D)
            if selected_hs.shape[0] == 0:
                selected_hs = hs[b][:1]

            # Pool and project each image scale
            projected_imgs = []
            for feat, pool, proj in zip(image_features, self.img_pools, self.img_proj):
                pooled = pool(feat[b:b+1]).squeeze()  # (C,)
                projected = proj(pooled)             # (D,)
                projected_imgs.append(projected)

            # Stack region + image features
            tokens = torch.cat([selected_hs, torch.stack(projected_imgs, dim=0)], dim=0)  # (M+4, D)

            # Self-Attention
            tokens = tokens.unsqueeze(1)  # (L, 1, D)
            encoded = self.encoder(tokens)  # (L, 1, D)
            cls_token = encoded.mean(dim=0).squeeze(0)  # (D,)

            logits = self.classifier(cls_token)
            logits_list.append(logits)

        logits = torch.stack(logits_list, dim=0)  # (B, num_classes)

        output = {
            "pred_logits": logits,
            "batched_input": batched_input,
        }

        return output


    def freeze_backbone(self):
        for param in self.detr_backbone.parameters():
            param.requires_grad = False
        self.detr_backbone.eval()


def build_hecto(args):
    model = safe_init(Hecto, args)
    model = load_model(model, args.ckpt_path)

    return model