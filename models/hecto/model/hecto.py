import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import load_model, safe_init, visualize
from models.hecto.model.encoder import HectoEncoder

class Hecto(nn.Module):
    def __init__(self, detr_backbone, device="cuda", num_classes=396):
        super().__init__()
        self.device = device
        self.detr_backbone = detr_backbone

        # encoder
        self.input_dim = 256
        self.hidden_dim = 512
        self.encoder = HectoEncoder(dim=self.hidden_dim, depth=4, heads=8, dropout=0.1)

        # query
        self.num_of_query = 4
        self.num_of_prompt = 6
        self.threshold = 0.2
        self.query_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.learnable_query = nn.Parameter(torch.randn(self.num_of_query, self.hidden_dim))
        nn.init.normal_(self.learnable_query, std=1.0)

        # context
        self.num_of_context = 64
        self.scale_of_context = 4
        self.context_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.context_down = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1),  # downsampling
            nn.GELU()
        )
        self.context_pool = nn.AdaptiveAvgPool1d(self.num_of_context)
        self.context_norm = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.scale_of_context)
        ])


        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes)
        )
        nn.init.normal_(self.classifier[-1].weight, std=0.1)

        self.query_count = 0
        self.batch_count = 0
        self.freeze_backbone()

    def forward(self, batched_input):
        with torch.no_grad():
            self.detr_backbone.eval()
            detr_output = self.detr_backbone(batched_input)

            pred_logits = detr_output["pred_logits"].sigmoid()   # (B, Q, 128)
            pred_boxes = detr_output["pred_boxes"]               # (B, Q, 4)
            hs = detr_output["hs"]                               # (B, Q, D)
            image_features = detr_output["image_feature"]        # list of 4 x [B, C, H, W]

        B, Q, D = hs.shape
        selected_hs_list = []
        query_padding_mask = []

        for b in range(B):
            pred_l = pred_logits[b]
            pred_b = pred_boxes[b]
            hs_b = hs[b]

            max_pred_l = pred_l.max(dim=-1)[0]
            mask = max_pred_l > self.threshold

            if mask.sum() == 0:
                selected = self.query_proj(hs_b[:1])  # fallback
            else:
                hs_sel = hs_b[mask]  # (N, D)
                box_sel = pred_b[mask]  # (N, 4)
                box_pos = self.box_positional_encoding(box_sel)  # (N, D)
                selected = self.query_proj(hs_sel) + box_pos  # (N, D)

            # visualize(pred_l[mask], pred_b[mask], "car . wheel . headlight . emblem . side mirror . window .", image=batched_input[0]['image'],is_logit=False,)

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
        query_tensor = self.query_norm(query_tensor)
        query_mask = torch.stack(padded_mask, dim=0)     # (B, max_len)

        context_list = []
        for scale, feat in enumerate(image_features):
            # B, C, H, W = lvl.shape
            tokens = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            projected = self.context_proj(tokens).transpose(1, 2)  # (B, D, N)
            projected_down = self.context_down(projected)
            pooled = self.context_pool(projected_down).transpose(1, 2)  # (B, 64, D)
            pooled = self.context_norm[scale](pooled)
            context_list.append(pooled)

        attended = self.encoder(query_tensor, context_list, query_mask)  # (B, max_len, D)

        learnable_attended = attended[:, :self.num_of_query]  # (B, 4, D)
        query_logits = self.classifier(learnable_attended)  # (B, 4, num_classes)
        logits = query_logits.mean(dim=1)  # (B, num_classes)

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

    def box_positional_encoding(self, boxes):
        """
        boxes: (N, 4) in normalized format [cx, cy, w, h]
        returns: (N, dim)
        """
        scales = torch.arange(self.hidden_dim // 8, device=boxes.device).float()
        scales = 10000 ** (2 * (scales // 2) / (self.hidden_dim // 4))

        encodings = []
        for i in range(4):
            x = boxes[:, i].unsqueeze(-1) / scales
            sin = torch.sin(x)
            cos = torch.cos(x)
            enc = torch.cat([sin, cos], dim=-1)
            encodings.append(enc)
        return torch.cat(encodings, dim=-1)  # (N, dim)


def build_hecto(args):
    model = safe_init(Hecto, args)
    model = load_model(model, args.ckpt_path)
    return model