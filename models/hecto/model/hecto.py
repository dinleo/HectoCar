import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import load_model, safe_init

class Hecto(nn.Module):
    def __init__(self, detr_backbone, device="cuda", num_classes=396):
        super().__init__()
        self.device = device
        self.threshold = 0.3
        self.parts_num = 6
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

        pred_logits = detr_output["pred_logits"].sigmoid()   # (B, Q, 128)
        hs = detr_output["hs"]                               # (B, Q, D)
        image_features = detr_output["image_feature"]        # list of 4 x [B, C_i, H, W]

        B, Q, D = hs.shape
        logits_list = []

        all_tokens = []
        key_padding_masks = []

        for b in range(B):
            pred = pred_logits[b]  # (Q, 128)
            hs_b = hs[b]  # (Q, D)
            mask = (pred > self.threshold)  # (Q, 128)

            class_hs_list = []
            for c in range(self.parts_num):
                cls_mask = mask[:, c]  # (Q,)
                if cls_mask.sum() == 0:
                    continue  # 이 클래스에 해당하는 region 없음
                avg = hs_b[cls_mask].mean(dim=0)  # (D,)
                class_hs_list.append(avg)

            if len(class_hs_list) == 0:
                selected_hs = hs_b[:1]
            else:
                selected_hs = torch.stack(class_hs_list, dim=0)  # (C′, D)

            # Pool and project each image scale
            projected_imgs = []
            for feat, pool, proj in zip(image_features, self.img_pools, self.img_proj):
                pooled = pool(feat[b:b+1]).squeeze()  # (C,)
                projected = proj(pooled)             # (D,)
                projected_imgs.append(projected)

            # Stack region + image features
            tokens = torch.cat([selected_hs, torch.stack(projected_imgs, dim=0)], dim=0)  # (M+4, D)
            all_tokens.append(tokens)

        L_max = max([t.shape[0] for t in all_tokens])
        padded_tokens = []
        for t in all_tokens:
            pad_len = L_max - t.shape[0]
            padded = F.pad(t, (0, 0, 0, pad_len))  # pad rows
            padded_tokens.append(padded)

        tokens_batch = torch.stack(padded_tokens, dim=1)  # (L_max, B, D)
        padding_mask = torch.tensor([
            [False] * t.shape[0] + [True] * (L_max - t.shape[0]) for t in all_tokens
        ], device=hs.device)  # (B, L_max)

        encoded = self.encoder(tokens_batch, src_key_padding_mask=padding_mask)
        encoded = encoded.transpose(0, 1)

        valid_mask = ~padding_mask  # (B, L_max)
        lengths = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # prevent divide-by-zero
        masked_encoded = encoded * valid_mask.unsqueeze(-1)  # (B, L_max, D)
        cls_token = masked_encoded.sum(dim=1) / lengths  # (B, D)

        logits = self.classifier(cls_token)

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