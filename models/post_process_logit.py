import torch
from detectron2.structures import Boxes, Instances
from torchvision.ops import box_convert

class PostProcessLogit(torch.nn.Module):
    """
    This module converts the model's output's (256 token logit) into (num of classes in prompt logit)
    ex: captions = "person . dog . cat ." [B, Q, 256] -> [B, Q, 3]
    """

    def __init__(self, tokenlizer=None, num_coco_cls=80, max_token_len=256, topk_num=300) -> None:
        super().__init__()
        self.tokenlizer = tokenlizer
        self.num_coco_cls = num_coco_cls
        self.max_token_len = max_token_len
        self.max_classes = max_token_len // 2
        self.topk_num = topk_num
        self.prompt_map_hub = {}
        self.coco_map_hub = {}
        self.cat2id = cat2id
        self.id2cat = id2cat

    def forward(self, logit, captions, mapping=None):
        """
        - Convert token-level logits (Batch, Query, Token=256) into class-level logits (Batch, Query, Class=128)
            - Max Class is 128 and pad with -inf for not present in the caption.
        - For classes composed of multiple tokens (e.g., "dining table"), their logits are averaged across tokens.
        - Alternatively, if you're working with BERT embeddings of shape (Batch, Token, Dim=768),
          you can apply `.transpose(1, 2)` and fed it to get class embeddings (Batch, Dim, Class)
          then, you can apply `output.transpose(1, 2)` to get (Batch, Class, Dim) for classification.
        Args:
            logit: raw logits (non-sigmoid), shape [B, Q, T]
                Token must self.max_token_len long if shorter, will be padded with -inf
            captions: list of caption strings like ["person . dog . cat .", "..." ]
            mapping:
                if set to "coco", converts (B, Q, C) to (B, Q, 80) for COCO evaluation,
                keeping only prompt classes that exactly match official COCO class names.
        Returns:
            logits_cls: Tensor of shape [B, Q, C]
        """
        assert logit.ndim == 3
        assert type(captions) == list
        logit = self.pad_token_dim(logit)
        logit[torch.isinf(logit)] = -20 # [B, Q, T]
        device = logit.device
        pos_maps = []

        for caption in captions:
            pos_map = self.get_token2prompt_map(caption)
            pos_maps.append(pos_map)

        pos_maps = torch.stack(pos_maps, dim=0).to(device)  # [B, C, T]
        # (B, Query, Token) @ (B, Token, Class) -> (B, Q, C)
        class_logit = logit @ pos_maps
        if mapping == 'coco':
            pos_maps = []
            for caption in captions:
                pos_map = self.get_prompt2coco_map(caption)
                pos_maps.append(pos_map)

            pos_maps = torch.stack(pos_maps, dim=0).to(device)
            # (B, Q, C) -> (B, Q, 80)
            class_logit = class_logit @ pos_maps

        pos_map_mask = (pos_maps.sum(dim=1) == 0)
        class_logit = class_logit.masked_fill(pos_map_mask.unsqueeze(1), float('-inf'))

        return class_logit

    def get_token2prompt_map(self, caption):
        """
        Create a [Token, C_prompt] tensor that maps each Token → C_prompt category index.
        Args:
            caption (str): raw caption string (e.g., "bear . zebra . giraffe .")
        Returns:
            Tensor: token_to_class_map [T, C_prompt]
        """
        if self.prompt_map_hub.get(caption, None) is not None:
            return self.prompt_map_hub[caption]

        tokenized = self.tokenlizer(caption, return_offsets_mapping=True, padding="longest", max_length=self.max_token_len)
        offsets = tokenized["offset_mapping"]
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        cat_names = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
        positive_map = torch.zeros((self.max_classes, self.max_token_len), dtype=torch.float)

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
                positive_map[i, j] = 1.0

            curr_char_pos = cat_end

        class_token_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-6) # [Class# in prompt, T]
        class_token_map = class_token_map.transpose(0, 1)
        self.prompt_map_hub[caption] = class_token_map
        return class_token_map

    def get_prompt2coco_map(self, caption):
        """
        Create a [C_prompt, 80] tensor that maps each prompt class → COCO category index.
        Returns:
            Tensor of shape [C_prompt, 80]
        """
        if caption in self.coco_map_hub:
            return self.coco_map_hub[caption]

        cat_names = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
        class_coco_map = torch.zeros(self.max_classes, self.num_coco_cls, dtype=torch.float)

        for i, name in enumerate(cat_names):
            coco_id = self.get_coco_id_from_name(name)
            if 0 <= coco_id < self.num_coco_cls:
                class_coco_map[i, coco_id] = 1.0

        self.coco_map_hub[caption] = class_coco_map
        return class_coco_map

    def select_topk(self, output, image_sizes):
        """
        Arguments:
            output have keys "pred_logits" and "pred_boxes"
                pred_logits (Tensor): tensor of shape (batch_size, num_queries, K).
                    The tensor predicts the classification probability for each query.
                pred_boxes (Tensor): tensors of shape (batch_size, num_queries, 4).
                    The tensor predicts 4-vector (x,y,w,h) box
                    regression values for every queryx
                image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        assert len(box_cls) == len(image_sizes)
        results = []

        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.topk_num, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
                zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_convert(box_pred_per_image, in_fmt="cxcywh", out_fmt="xyxy"))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
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

    def get_all_captions(self):
        return ". ".join(self.cat2id.keys()) + "."

    def get_all_classes(self):
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
official_coco_idx = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90
}
