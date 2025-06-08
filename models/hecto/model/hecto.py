import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from models.model_utils import load_model, safe_init, visualize
from models.hecto.model.base_model import BaseModel, BertConfigW
from models.hecto.model.encoder import HectoEncoder
from models.caption_processor import CaptionProcessor

from PIL import Image
from transformers import AutoFeatureExtractor, SwinModel
import timm


class Hecto(nn.Module):
    def __init__(self,
                 detr_backbone,
                 region_size=256,
                 image_size=1024,
                 image_scale=3,
                 query_size=768,
                 context_size=768,
                 num_query_token=16,
                 num_context_token=32,
                 num_classes=396,
                 device="cuda",
                 ):
        super().__init__()
        self.device = device
        self.detr_backbone = detr_backbone
        self.detr_backbone.captions = "car . wheel . headlight . taillight . emblem . side mirror . window . grille . door . bumper . windshield . tire . roof . trunk ."
        self.caption_processor = CaptionProcessor(self.detr_backbone.tokenizer)
        self.image_backbone_name = "microsoft/swin-base-patch4-window12-384-in22k"
        self.image_backbone = SwinModel.from_pretrained(self.image_backbone_name).eval().to("cuda")
        self.extractor = AutoFeatureExtractor.from_pretrained(self.image_backbone_name)

        # target
        self.target_csv = "models/hecto/class_names.csv"
        self.target_embedding = None
        self.target_names = []
        self.target_ids = {}

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
        # self.num_of_prompt = num_of_prompt
        # self.num_of_query = num_of_prompt + 1
        self.learnable_query = nn.Parameter(torch.randn(1, self.query_size))
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
        # visualize("model", detr_output, batched_input[0]["image"], caption=self.detr_backbone.captions)

        # 2. Make Query Input
        B, Q, D = hs.shape
        query_list = []
        for b in range(B):
            logit_b = pred_logits[b].sigmoid()
            boxes_b = pred_boxes[b]
            hs_b = hs[b]
            mask = logit_b.max(dim=-1)[0] > self.threshold

            if mask.sum() == 0:
                hs_sel = hs_b[0]
            else:
                hs_sel = hs_b[mask]        # (N, D)
                logit_sel = logit_b[mask]  # (N, 128)
                box_sel = boxes_b[mask]    # (N, 4)

            prompt_query = self.query_proj(hs_sel)  # (N, hidden_dim)
            cls_query = self.learnable_query
            query_b = torch.cat([cls_query, prompt_query], dim=0)  # (1 + N, hidden_dim)
            query_list.append(query_b)
            self.batch_count += 1
            self.query_count += len(query_b)

        # Padding
        max_len = max(q.shape[0] for q in query_list)
        for i in range(B):
            q = query_list[i]
            pad_len = max_len - q.shape[0]
            if pad_len > 0:
                pad_q = torch.zeros(pad_len, self.query_size, device=q.device)
                query_list[i] = torch.cat([q, pad_q], dim=0)

        query_tokens = torch.stack(query_list, dim=0)  # (B, max_len, D)
        valid_mask = (query_tokens.abs().sum(dim=-1) > 0).bool()   # (B, max_len)
        query_tokens = self.query_norm(query_tokens)

        # Make Context Input
        ms_features = self.get_image_feature(batched_input)
        ms_context_tokens = []
        for s in range(len(ms_features)):
            feat = ms_features[s]  # (B, D, H, W)
            ctx = self.context_projection[s](feat)
            ctx = ctx.flatten(2).transpose(1, 2).contiguous()  # (B, T, D)
            ctx = self.context_tokens_ln[s](ctx)
            ms_context_tokens.append(ctx)

        # Encoder
        encoder_output = self.encoder(
            query_tokens=query_tokens,
            ms_context_tokens=ms_context_tokens,
            query_mask=valid_mask,
        )

        targets = self.make_targets(batched_input)
        last_hidden_state = encoder_output['last_hidden_state']  # (B, Q, D)
        cls_token = last_hidden_state[:, 0]       # (B, D)

        prompt_token = last_hidden_state[:, 1:, :]   # (B, Q-1, D)
        prompt_mask = valid_mask[:, 1:].float().unsqueeze(-1) # (B, Q-1, 1)
        prompt_token = prompt_token * prompt_mask
        mask_sum = prompt_mask.sum(dim=1).clamp(min=1e-6)
        prompt_token = prompt_token.sum(dim=1) / mask_sum

        pred_logits = self.classifier(cls_token + prompt_token)  # (B, num_classes)
        return {
            "pred_logits": pred_logits,
            "query_pred": cls_token,
            "batched_input": batched_input,
            "targets": targets,
        }

    def init_encoder(self):
        encoder_config = BertConfigW()
        encoder_config.region_size = self.query_size
        encoder_config.query_size = self.query_size
        encoder_config.context_size = self.context_size
        encoder_config.num_context_token = self.num_context_token
        encoder_config.num_hidden_layers = self.num_hidden_layers

        return HectoEncoder(encoder_config)

    def get_image_feature(self, batched_input):
        image_paths = [b["file_name"] for b in batched_input]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.extractor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.image_backbone(pixel_values=pixel_values)

        last_hidden_state = outputs['last_hidden_state']
        B, N, D = last_hidden_state.shape
        H = W = int(N ** 0.5)  # 12 x 12 assumed
        feat_map = last_hidden_state.transpose(1, 2).reshape(B, D, H, W)
        ms_features = []
        feat = feat_map
        for i in range(self.image_scale):
            if i == 0:
                pooled = feat  # original
            else:
                kernel = 2 ** i
                pooled = F.avg_pool2d(feat_map, kernel_size=kernel, stride=kernel)  # downscale

            ms_features.append(pooled)  # shape: [B, D, H', W']

        return ms_features

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

    def embed_words_from_csv(self):
        df = pd.read_csv(self.target_csv)
        class_names = df["CarName"].str.strip("'").tolist()
        class_ids = {name.lower(): idx for idx, name in enumerate(class_names)}

        embeddings = []
        chunk_size = 8
        print(f"[INIT] Generating target embeddings from {self.target_csv}...")
        for i in tqdm(range(0, len(class_names), chunk_size)):
            chunk_names = class_names[i:i + chunk_size]
            caption = [".".join(chunk_names) + "."]
            with torch.no_grad():
                output = self.detr_backbone.text_backbone(caption)
                bert_output = output["bert_output"].transpose(1, 2)  # [B, 768, T]
                emb = self.caption_processor(bert_output, caption, return_dot=False)
                emb = emb.transpose(1, 2)[0].to("cpu")  # [C, 768]
                embeddings.append(emb[:len(chunk_names)])
                assert emb[len(chunk_names)][0] == float("-inf")

        self.target_embedding = torch.cat(embeddings, dim=0)
        self.target_names = class_names
        self.target_ids = class_ids

    def make_targets(self, batched_input):
        """
        Build target embeddings and labels for matched region queries.
        Returns:
            Dict with:
                - 'query_target': [B, D] main word embeddings.
                - 'target_labels': [B,] class indices
                - 'target_names': word names for logging/debug.
        """
        batch_size = len(batched_input)

        # Make target
        query_target = torch.zeros(batch_size, self.query_size)
        target_labels = torch.zeros(batch_size)
        target_names = []
        for b in range(batch_size):
            label = batched_input[b]['label']
            query_target[b] = self.target_embedding[label]
            target_labels[b] = label
            target_names.append(self.target_names[label])
        targets = {
            "query_target" : query_target.to(self.device),
            "target_labels" : target_labels.to(self.device),
            "target_names" : target_names,
        }
        return targets

    def prepare(self):
        self.embed_words_from_csv()

    def print(self):
        if self.batch_count == 0:
            self.batch_count = 1
        print(f"Query Count: {self.query_count}")
        print(f"Batch Count: {self.batch_count}")
        print(f"Average : {self.query_count / self.batch_count:.2f}")

def build_hecto(args):
    model = safe_init(Hecto, args)
    model = load_model(model, args.ckpt_path)
    return model