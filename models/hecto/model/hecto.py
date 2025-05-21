import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import load_model, safe_init
from models.hecto.model.encoder import HectoEncoder

class Hecto(nn.Module):
    def __init__(self, detr_backbone, device="cuda", num_classes=396):
        super().__init__()
        self.device = device
        self.threshold = 0.3
        self.num_of_query = 6
        self.num_of_context = 128
        self.detr_backbone = detr_backbone

        self.encoder = HectoEncoder(dim=256, depth=4, heads=8, dropout=0.1)
        self.img_proj = nn.Linear(256, 256)
        self.img_pool = nn.AdaptiveAvgPool1d(self.num_of_context)  # reduce to 64 tokens per scale

        self.learnable_query = nn.Parameter(torch.randn(self.num_of_query, 256))

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
            image_features = detr_output["image_feature"]        # list of 4 x [B, C, H, W]

        B, Q, D = hs.shape
        selected_hs_list = []
        query_padding_mask = []

        for b in range(B):
            pred = pred_logits[b]
            hs_b = hs[b]
            mask = (pred > self.threshold)

            class_hs_list = []
            for c in range(self.num_of_query):
                cls_mask = mask[:, c]
                if cls_mask.sum() == 0:
                    continue
                avg = hs_b[cls_mask].mean(dim=0)
                class_hs_list.append(avg)

            if len(class_hs_list) == 0:
                selected = hs_b[:1]
            else:
                selected = torch.stack(class_hs_list, dim=0)

            combined_query = torch.cat([self.learnable_query.to(selected.device), selected], dim=0)
            selected_hs_list.append(combined_query)
            query_padding_mask.append(torch.zeros(combined_query.size(0), dtype=torch.bool, device=hs.device))

        max_len = max(t.size(0) for t in selected_hs_list)
        padded_query = []
        padded_mask = []
        for q, m in zip(selected_hs_list, query_padding_mask):
            pad_len = max_len - q.size(0)
            padded_query.append(F.pad(q, (0, 0, 0, pad_len)))
            padded_mask.append(F.pad(m, (0, pad_len), value=True))

        query_tensor = torch.stack(padded_query, dim=0)  # (B, max_len, D)
        query_mask = torch.stack(padded_mask, dim=0)     # (B, max_len)

        context_list = []
        for lvl in image_features:
            # B, C, H, W = lvl.shape
            tokens = lvl.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            projected = self.img_proj(tokens).transpose(1, 2)  # (B, D, N)
            pooled = self.img_pool(projected).transpose(1, 2)  # (B, 64, D)
            context_list.append(pooled)

        attended = self.encoder(query_tensor, context_list, query_mask)  # (B, max_len, D)
        attended = attended.masked_fill(query_mask.unsqueeze(-1), 0.0)
        lengths = (~query_mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = attended.sum(dim=1) / lengths

        logits = self.classifier(pooled)

        return {
            "pred_logits": logits,
            "batched_input": batched_input,
        }

    def freeze_backbone(self):
        for param in self.detr_backbone.parameters():
            param.requires_grad = False
        self.detr_backbone.eval()


def build_hecto(args):
    model = safe_init(Hecto, args)
    model = load_model(model, args.ckpt_path)
    return model