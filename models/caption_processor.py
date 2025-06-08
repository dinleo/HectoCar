import torch
from detectron2.structures import Boxes, Instances
from torchvision.ops import box_convert

class CaptionProcessor(torch.nn.Module):
    """
    This module converts the model's output's (256 Prompt token level logit) into (Num of Classes in prompt logit)
    ex: captions = "person . dog . wine glasses ." [B, Q, 256] -> [B, Q, 3]

    Instead of using scatter operations to accumulate and average token-level logits into class-level logits,
    this module uses batched matrix multiplication (matmul) with a precomputed soft mapping matrix.

    This approach:
    - Supports soft token-to-class mappings (e.g., multi-token classes like "dining table")
    - Enables efficient batched computation on GPU
    - Produces gradients that are smoother and easier to optimize than hard indexing (e.g., scatter)
    """

    def __init__(self, tokenlizer=None, num_coco_cls=80, max_token_len=256, topk_num=300, use_map_cache=False) -> None:
        super().__init__()
        self.tokenlizer = tokenlizer
        self.num_coco_cls = num_coco_cls
        self.max_token_len = max_token_len
        self.max_classes = max_token_len // 2
        self.topk_num = topk_num
        self.use_map_cache = use_map_cache

        self.prompt_map_hub = {}
        self.dot_map_hub = {}
        self.coco_map_hub = {}
        self.cat2id = cat2id
        self.id2cat = id2cat

    def forward(self, logits, captions, return_dot=False):
        """
        - Convert token-level logits (Batch, Query, Token=256) into class-level logits (Batch, Query, Class=128)
            - Max Class is 128 and pad with -inf for rest.
        - For classes composed of multiple tokens (e.g., "dining table"), their logits are averaged across 3 tokens (including space token).
        - Alternatively, if you're working with BERT embeddings of shape (Batch, Token, Dim=768),
          you can apply `.transpose(1, 2)` and fed it to get class embeddings (Batch, Dim, Class)
          then, you can apply `output.transpose(1, 2)` to get (Batch, Class, Dim) for classification.
        Args:
            logits: raw logits (non-sigmoid), shape [B, Q, T]
                Token must self.max_token_len long if shorter, will be padded with -inf
            captions: list of caption strings like ["person . dog . cat .", "..." ]
            return_dot: if True, return not only class_logit, but also dot_logit(seperator logit)
        Returns:
            class_logit: Tensor of shape [B, Q, C]
            [class_logit, dot_logit]: When return_dot is True
        """
        assert logits.ndim == 3
        assert type(captions) == list
        B, Q, T = logits.size()
        device = logits.device

        row_mask = logits.isinf().all(dim=2)  # (B, Q)

        # Prevent nan
        logits[torch.isinf(logits)] = -20 # [B, Q, T]

        # Build Maps
        class_maps = [self.get_token2prompt_map(c)[:T] for c in captions]
        class_maps = torch.stack(class_maps, dim=0).to(device)  # [B, C, T]

        # (B, Query, Token) @ (B, Token, Class) -> (B, Q, C)
        class_logit = logits @ class_maps

        col_mask = (class_maps.sum(dim=1) == 0)
        class_logit = class_logit.masked_fill(col_mask.unsqueeze(1), float('-inf'))
        class_logit = class_logit.masked_fill(row_mask.unsqueeze(2), float('-inf'))

        if not return_dot:
            return class_logit
        else:
            dot_maps = [self.get_token2dot_map(c)[:T] for c in captions]
            dot_maps = torch.stack(dot_maps, dim=0).to(device)  # [B, C, T]
            dot_logit = logits @ dot_maps

            col_mask = (dot_maps.sum(dim=1) == 0)
            dot_logit = dot_logit.masked_fill(col_mask.unsqueeze(1), float('-inf'))
            dot_logit = dot_logit.masked_fill(row_mask.unsqueeze(2), float('-inf'))

            return class_logit, dot_logit

    def convert_coco(self, logits, captions):
        """
        converts (B, Q, C) to (B, Q, 80) for COCO evaluation,
        Only prompt classes that exactly match the official COCO class names are used
        (e.g., "Dog" and "dog" are considered different and do not match).

        Classes Column not present in the prompt will be filled with '-inf'.
        """
        assert logits.ndim == 3
        assert type(captions) == list
        device = logits.device
        B, Q, C = logits.size()
        row_mask = logits.isinf().all(dim=2)

        # prevent nan
        logits[torch.isinf(logits)] = -20

        coco_maps = [self.get_prompt2coco_map(caption)[:C] for caption in captions]
        coco_maps = torch.stack(coco_maps, dim=0).to(device)

        # (B, Q, C) @ (B, C, 80) -> (B, Q, 80)
        coco_logits = logits @ coco_maps

        col_mask = (coco_maps.sum(dim=1) == 0)
        coco_logits = coco_logits.masked_fill(col_mask.unsqueeze(1), float('-inf'))
        coco_logits = coco_logits.masked_fill(row_mask.unsqueeze(2), float('-inf'))

        return coco_logits

    def get_token2prompt_map(self, caption):
        """
        Create a [Token, C_prompt] tensor that maps each Token → C_prompt category index.
        Args:
            caption (str): raw caption string (e.g., "bear . zebra . giraffe .")
        Returns:
            Tensor: token_to_class_map [T, C_prompt]
        """
        if self.use_map_cache and caption in self.prompt_map_hub:
            return self.prompt_map_hub[caption]

        tokenized = self.tokenlizer(caption, return_offsets_mapping=True, padding="longest", max_length=self.max_token_len)
        offsets = tokenized["offset_mapping"]
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        cat_names = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
        class_token_map = torch.zeros((self.max_classes, self.max_token_len), dtype=torch.float)

        curr_char_pos = 0
        for i, cat in enumerate(cat_names):
            try:
                cat_start = caption.index(cat, curr_char_pos)
                cat_end = cat_start + len(cat)
            except ValueError:
                continue

            for j, (start, end) in enumerate(offsets):
                if start is None or end is None:
                    continue
                if end <= cat_start:
                    continue
                if start >= cat_end:
                    break
                class_token_map[i, j] = 1.0

            curr_char_pos = cat_end

        class_token_map = class_token_map / (class_token_map.sum(-1)[:, None] + 1e-6) # [Class# in prompt, T]
        class_token_map = class_token_map.transpose(0, 1)
        if self.use_map_cache:
            self.prompt_map_hub[caption] = class_token_map
        return class_token_map

    def get_token2dot_map(self, caption):
        """
        Create a [Token, C_prompt] tensor that maps each class's '.' separator token → class index.
        For each class name, finds the next '.' in the text and marks the corresponding token index.

        Returns:
            dot_token_map: [T, C_prompt] where each column i has a 1.0 at the token that corresponds to the dot after class i.
        """
        if self.use_map_cache and caption in self.dot_map_hub:
            return self.dot_map_hub[caption]

        tokenized = self.tokenlizer(caption, return_offsets_mapping=True, padding="longest",
                                    max_length=self.max_token_len)
        offsets = tokenized["offset_mapping"]
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        cat_names = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
        dot_token_map = torch.zeros((self.max_classes, self.max_token_len), dtype=torch.float)

        curr_char_pos = 0
        for i, cat in enumerate(cat_names):
            try:
                cat_start = caption.index(cat, curr_char_pos)
                cat_end = cat_start + len(cat)
                dot_pos = caption.index(".", cat_end)  # find dot after this category
            except ValueError:
                continue

            # find the token index that contains this dot character
            for j, (start, end) in enumerate(offsets):
                if start is None or end is None:
                    continue
                if start <= dot_pos < end:
                    dot_token_map[i, j] = 1.0
                    break

            curr_char_pos = dot_pos + 1  # move cursor to after the dot

        # Normalize to ensure it's a valid attention-style map
        dot_token_map = dot_token_map / (dot_token_map.sum(-1, keepdim=True) + 1e-6)  # [C, T]
        dot_token_map = dot_token_map.transpose(0, 1)  # → [T, C]
        if self.use_map_cache:
            self.dot_map_hub[caption] = dot_token_map
        return dot_token_map

    def get_prompt2coco_map(self, caption):
        """
        Create a [C_prompt, 80] tensor that maps each prompt class → COCO category index.
        Returns:
            Tensor of shape [C_prompt, 80]
        """
        if self.use_map_cache and caption in self.coco_map_hub:
            return self.coco_map_hub[caption]

        cat_names = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
        class_coco_map = torch.zeros(self.max_classes, self.num_coco_cls, dtype=torch.float)

        for i, name in enumerate(cat_names):
            coco_id = self.get_coco_id_from_name(name)
            if 0 <= coco_id:
                class_coco_map[i, coco_id] = 1.0

        if self.use_map_cache:
            self.coco_map_hub[caption] = class_coco_map
        return class_coco_map

    def select_topk(self, output, image_sizes, is_logit=True, threshold=0):
        """
        Select top-K predictions per image based on class confidence scores.
        Args:
            output (dict): Contains:
                - "pred_logits" (Tensor): [B, Q, C], class scores (logits or probabilities)
                - "pred_boxes"  (Tensor): [B, Q, 4], predicted boxes in (cx, cy, w, h) format
            image_sizes (List[Tuple[int, int]]): Original (height, width) of each image.
            is_logit (bool): If True, apply sigmoid to pred_logits before scoring.
            threshold (float): Confidence threshold(for prob not logits) for filtering predictions.
        Returns:
            List[Instances]: Per-image detection results with fields:
                - pred_boxes
                - scores
                - pred_classes
        """
        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        assert len(pred_logits) == len(image_sizes)
        B, Q, C = pred_logits.shape
        K = min(self.topk_num, Q)
        if is_logit:
            pred_logits = pred_logits.sigmoid()

        topk_values, topk_indexes = torch.topk(pred_logits.view(B, -1), K, dim=1)
        q_idx = topk_indexes // C
        c_idx = topk_indexes % C
        boxes = torch.gather(pred_boxes, 1, q_idx.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        results = []
        for i in range(B):
            score_i = topk_values[i]
            box_i = boxes[i]
            label_i = c_idx[i]
            mask = score_i > threshold

            result = Instances(image_sizes[i])
            result.pred_boxes = Boxes(box_convert(box_i[mask], in_fmt="cxcywh", out_fmt="xyxy"))
            result.pred_boxes.scale(scale_x=image_sizes[i][1], scale_y=image_sizes[i][0])
            result.scores = score_i[mask]
            result.pred_classes = label_i[mask]
            results.append(result)

        return results

    def pad_token_dim(self, x):
        if x.shape[2] == self.max_token_len:
            return x
        elif x.shape[2] > self.max_token_len:
            raise ValueError(f"Input token dim {x.shape[-1]} exceeds max_token_len {self.max_token_len}")

        pad_width = self.max_token_len - x.shape[-1]
        pad_shape = list(x.shape[:-1]) + [pad_width]
        pad = torch.full(pad_shape, float("-inf"), dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    def get_coco_id_from_name(self, name):
        return self.cat2id.get(name, -1)

    def get_coco_name_from_id(self, id_):
        return self.id2cat.get(id_, "unknown")

    def get_coco_all_captions(self):
        return ". ".join(self.cat2id.keys()) + "."

    def get_coco_all_classes(self):
        return list(self.cat2id.keys())

cat2id = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79
}
id2cat = {v: k for k, v in cat2id.items()}
# official_coco_idx = {
#     "person": 1,
#     "bicycle": 2,
#     "car": 3,
#     "motorcycle": 4,
#     "airplane": 5,
#     "bus": 6,
#     "train": 7,
#     "truck": 8,
#     "boat": 9,
#     "traffic light": 10,
#     "fire hydrant": 11,
#     "stop sign": 13,
#     "parking meter": 14,
#     "bench": 15,
#     "bird": 16,
#     "cat": 17,
#     "dog": 18,
#     "horse": 19,
#     "sheep": 20,
#     "cow": 21,
#     "elephant": 22,
#     "bear": 23,
#     "zebra": 24,
#     "giraffe": 25,
#     "backpack": 27,
#     "umbrella": 28,
#     "handbag": 31,
#     "tie": 32,
#     "suitcase": 33,
#     "frisbee": 34,
#     "skis": 35,
#     "snowboard": 36,
#     "sports ball": 37,
#     "kite": 38,
#     "baseball bat": 39,
#     "baseball glove": 40,
#     "skateboard": 41,
#     "surfboard": 42,
#     "tennis racket": 43,
#     "bottle": 44,
#     "wine glass": 46,
#     "cup": 47,
#     "fork": 48,
#     "knife": 49,
#     "spoon": 50,
#     "bowl": 51,
#     "banana": 52,
#     "apple": 53,
#     "sandwich": 54,
#     "orange": 55,
#     "broccoli": 56,
#     "carrot": 57,
#     "hot dog": 58,
#     "pizza": 59,
#     "donut": 60,
#     "cake": 61,
#     "chair": 62,
#     "couch": 63,
#     "potted plant": 64,
#     "bed": 65,
#     "dining table": 67,
#     "toilet": 70,
#     "tv": 72,
#     "laptop": 73,
#     "mouse": 74,
#     "remote": 75,
#     "keyboard": 76,
#     "cell phone": 77,
#     "microwave": 78,
#     "oven": 79,
#     "toaster": 80,
#     "sink": 81,
#     "refrigerator": 82,
#     "book": 84,
#     "clock": 85,
#     "vase": 86,
#     "scissors": 87,
#     "teddy bear": 88,
#     "hair drier": 89,
#     "toothbrush": 90
# }
