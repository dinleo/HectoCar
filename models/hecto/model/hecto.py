import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import load_model, safe_init, visualize
from models.hecto.model.base_model import BaseModel, BertConfigW
from models.hecto.model.encoder import HectoEncoder

class Hecto(nn.Module):
    def __init__(self,
                 detr_backbone,
                 region_size=256,
                 image_size=256,
                 image_scale=4,
                 query_size=768,
                 context_size=768,
                 num_query_token=16,
                 num_context_token=32,
                 num_of_prompt=5,
                 num_classes=396,
                 device="cuda",
                 ):
        super().__init__()
        self.device = device
        self.detr_backbone = detr_backbone

        # Backbone
        self.region_size = region_size
        self.image_size = image_size
        self.image_scale = image_scale

        # Encoder
        self.query_size = query_size
        self.context_size = context_size
        self.num_query_token = num_query_token
        self.num_context_token = num_context_token
        self.num_hidden_layers = image_scale * 2
        # ex [(16, 16), (8, 8), (4, 4), (2, 2)]
        self.pool_sizes = [(max(1, self.num_context_token // (2 ** i)),) * 2 for i in range(self.image_scale)]
        self.encoder = self.init_encoder()

        # query
        self.num_of_prompt = num_of_prompt
        self.num_of_query = num_of_prompt + 1
        self.learnable_query = nn.Parameter(torch.randn(self.num_of_query, self.query_size))
        self.threshold = 0.2
        self.query_proj = nn.Linear(self.region_size, self.query_size)
        self.query_norm = nn.LayerNorm(self.query_size)

        # Input context
        self.context_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(image_size, context_size, kernel_size=1),
            )
            for _ in range(image_scale)
        ])
        self.context_tokens_ln = nn.ParameterList([nn.LayerNorm(context_size) for _ in range(image_scale)])

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.query_size, self.query_size),
            nn.ReLU(),
            nn.Linear(self.query_size, num_classes)
        )

        self.query_count = 0
        self.batch_count = 0
        self.freeze_backbone()

    def forward(self, batched_input):
        # 1. Backbone

        self.detr_backbone.eval()
        detr_output = self.detr_backbone(batched_input)

        pred_logits = detr_output["pred_logits"]             # (B, Q, 128)
        pred_boxes = detr_output["pred_boxes"]               # (B, Q, 4)
        hs = detr_output["hs"]                               # (B, Q, D)
        image_output = detr_output["image_output"]        # list of 4 x [B, C, H, W]
            # visualize("model", detr_output, batched_input[0]["image"], caption="wheel . headlight . emblem . side mirror . window .")
        # 2. Make Query Input
        B, Q, D = hs.shape
        query_list = []
        for b in range(B):
            logit_b = pred_logits[b].sigmoid()
            boxes_b = pred_boxes[b]
            hs_b = hs[b]
            mask = logit_b.max(dim=-1)[0] > self.threshold

            if mask.sum() == 0:
                prompt_hs = torch.zeros(self.num_of_prompt, self.query_size, device=hs.device)
            else:
                hs_sel = hs_b[mask]        # (N, D)
                logit_sel = logit_b[mask]  # (N, 128)
                box_sel = boxes_b[mask]    # (N, 4)

                prompt_hs = []
                for i in range(self.num_of_prompt):  # e.g., wheel, headlight, ...
                    logits_i = logit_sel[:, i]  # (N,)
                    mask_c = logits_i > self.threshold

                    if mask_c.sum() == 0:
                        pooled = torch.zeros(self.region_size, device=hs.device)
                    else:
                        pooled = hs_sel[mask_c].mean(dim=0)  # (D,)
                    prompt_hs.append(pooled)

                prompt_hs = torch.stack(prompt_hs, dim=0)  # (num_prompt, hidden_dim)
                prompt_hs = self.query_proj(prompt_hs)

            prompt_query = self.learnable_query[1:] + prompt_hs  # (num_prompt, hidden)
            cls_query = self.learnable_query[0].unsqueeze(0)   # (1, hidden)
            query_b = torch.cat([cls_query, prompt_query], dim=0)  # (num_prompt+1, hidden)
            query_list.append(query_b)

        query_tokens = torch.stack(query_list, dim=0)  # (B, max_len, D)
        query_tokens = self.query_norm(query_tokens)

        # Make Context Input
        _, _, ms_pos, ms_features, ms_mask = image_output
        ms_context_tokens = []
        for s in range(len(ms_features)):
            feat = ms_features[s]  # (B, C, H, W)
            pos = ms_pos[s]  # (B, C, H, W)
            mask = ms_mask[s]  # (B, H, W), True for pad

            # 1. mask → float form (0 = keep, 1 = pad)
            mask_f = mask.unsqueeze(1).float()  # (B, 1, H, W)

            # 2. mask-out padded region
            feat_with_pos = feat + pos
            feat_with_pos = feat_with_pos.masked_fill(mask_f.bool(), 0.0)

            # 3. projection
            ctx = self.context_projection[s](feat_with_pos)  # (B, D, H, W)

            # 4. pooling
            pooled = F.adaptive_avg_pool2d(ctx, output_size=self.pool_sizes[s])  # (B, D, h, w)
            mask_pooled = F.adaptive_avg_pool2d(1.0 - mask_f, output_size=self.pool_sizes[s])  # (B, 1, h, w)
            mask_pooled = mask_pooled.clamp(min=1e-6)  # divide-by-zero

            pooled = pooled / mask_pooled

            # 6. flatten & LN
            tokens = pooled.flatten(2).transpose(1, 2).contiguous()  # (B, T, D)
            tokens = self.context_tokens_ln[s](tokens)
            ms_context_tokens.append(tokens)

        # Encoder
        encoder_output = self.encoder(
            query_tokens=query_tokens,
            ms_context_tokens=ms_context_tokens
        )

        last_hidden_state = encoder_output['last_hidden_state']  # (B, Q, D)
        cls_logit = last_hidden_state[:, 0]       # (B, D)
        prompt_logit = last_hidden_state[:, 1:]   # (B, 5, D)
        prompt_logit = prompt_logit.mean(dim=1)
        logits = self.classifier(cls_logit + prompt_logit)  # (B, num_classes)

        return {
            "pred_logits": logits,
            "batched_input": batched_input,
        }

    def init_encoder(self):
        encoder_config = BertConfigW()
        encoder_config.region_size = self.query_size
        encoder_config.query_size = self.query_size
        encoder_config.context_size = self.context_size
        encoder_config.num_context_token = self.num_context_token
        encoder_config.num_hidden_layers = self.num_hidden_layers

        return HectoEncoder(encoder_config)

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
        scales = torch.arange(self.query_size // 8, device=boxes.device).float()
        scales = 10000 ** (2 * (scales // 2) / (self.query_size // 4))

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