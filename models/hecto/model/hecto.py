import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import load_model, safe_init
from models.hecto.model.encoder import HectoEncoder

class Hecto(nn.Module):
    def __init__(self, detr_backbone, device="cuda", num_classes=396):
        super().__init__()
        self.device = device
        self.threshold = 0.2
        self.num_of_query = 16
        self.num_of_context = 64
        self.num_of_prompt = 6
        self.detr_backbone = detr_backbone
        self.input_dim = 256
        self.hidden_dim = 512

        self.encoder = HectoEncoder(dim=self.hidden_dim, depth=4, heads=8, dropout=0.1)
        self.query_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.img_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.img_down = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1),  # downsampling
            nn.GELU()
        )
        self.img_pool = nn.AdaptiveAvgPool1d(self.num_of_context)

        self.learnable_query = nn.Parameter(torch.randn(self.num_of_query, self.hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes)
        )

        self.query_count = 0
        self.batch_count = 0
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
            selected_hs = []
            for c in range(self.num_of_prompt):  # C
                cls_mask = pred[:, c] > self.threshold  # (Q,)
                if cls_mask.any():
                    selected_hs.append(hs_b[cls_mask])  # (N_c, D)

            if len(selected_hs) == 0:
                selected = hs_b[:1]
            else:
                selected = torch.cat(selected_hs, dim=0)  # (N, D)
            selected = self.query_proj(selected)

            combined_query = torch.cat([self.learnable_query.to(selected.device), selected], dim=0)
            self.query_count += len(combined_query)
            self.batch_count += 1

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
            projected_down = self.img_down(projected)
            pooled = self.img_pool(projected_down).transpose(1, 2)  # (B, 64, D)
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
    def print_query_len(self):
        ave = self.query_count / self.batch_count
        print("Average query:" , ave)

    def freeze_backbone(self):
        for param in self.detr_backbone.parameters():
            param.requires_grad = False
        self.detr_backbone.eval()


def build_hecto(args):
    model = safe_init(Hecto, args)
    model = load_model(model, args.ckpt_path)
    return model