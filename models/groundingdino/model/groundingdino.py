# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List, Mapping, Any
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..util import get_tokenlizer
from ..util.box_ops import box_xyxy_to_cxcywh
from ..util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)

from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .utils import MLP, ContrastiveEmbed

from detectron2.modeling import detector_postprocess
from detectron2.structures import ImageList

from models.model_utils import safe_init, load_model, visualize
from models.caption_processor import CaptionProcessor


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [123.675, 116.280, 103.530],
        device="cuda"
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.device = device
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        self.captions = "wheel . headlight . emblem . side mirror . window ."

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)
        self.bert.to(self.device)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

        #
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        # coco
        self.post_logit = CaptionProcessor(self.tokenizer)

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def set_image_tensor(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        self.features, self.poss = self.backbone(samples)

    def unset_image_tensor(self):
        if hasattr(self, 'features'):
            del self.features
        if hasattr(self,'poss'):
            del self.poss 

    def set_image_features(self, features , poss):
        self.features = features
        self.poss = poss

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def text_backbone(self, captions, device="cuda"):
        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )
        if text_self_attention_masks.shape[1] > self.max_text_len:
            # raise f"Token Len: {text_self_attention_masks.shape[1]}"
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, tokens, 768
        text_token_mask = tokenized.attention_mask.bool()  # bs, tokens
        output = {
            'bert_output': bert_output['last_hidden_state'],
            'text_token_mask': text_token_mask,
            'position_ids': position_ids,
            'text_self_attention_masks': text_self_attention_masks,
            'cate_to_token_mask_list': cate_to_token_mask_list,
        }
        return output

    def image_backbone_output(self, batched_input):
        images = self.preprocess_image(batched_input)
        assert isinstance(images, ImageList)
        samples = nested_tensor_from_tensor_list(images)
        # image backbone
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        return images, features, poss, srcs, masks

    def forward(self, batched_input, hub=None):
        # prepare targets
        B = len(batched_input)
        captions = [self.captions] * B
        text_output = self.text_backbone(captions, self.device)
        image_output = self.image_backbone_output(batched_input)

        # Text
        bert_output = text_output["bert_output"]
        text_token_mask = text_output["text_token_mask"]
        position_ids = text_output["position_ids"]
        text_self_attention_masks = text_output["text_self_attention_masks"]
        cate_to_token_mask_list = text_output["cate_to_token_mask_list"]

        # Encode
        encoded_text = self.feat_map(bert_output)  # bs, tokens, d_model=256

        text_dict = {
            "encoded_text": encoded_text,  # bs, tokens, d_model
            "text_token_mask": text_token_mask,  # bs, tokens
            "position_ids": position_ids,  # bs, tokens
            "text_self_attention_masks": text_self_attention_masks,  # bs, tokens, tokens
        }

        # Image
        images, features, poss, srcs, masks = image_output

        # Decoder
        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        # 6 * [bs, 900, 256] [bs, 900, 4]
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        output_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs) # offset (hs 256 -> box 4)
            layer_output_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig) # prev box + offset
            layer_output_unsig = layer_output_unsig.sigmoid()
            output_coord_list.append(layer_output_unsig)
        output_coord_list = torch.stack(output_coord_list) # 6 * [1, 900, 4]

        # output
        output_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        out = {
            "pred_logits": output_class[-1],
            "pred_boxes": output_coord_list[-1],
            "hs": hs[-1],
            "image_output": image_output,
        }


        # change token logits to class logits
        out["pred_logits"] = self.post_logit(out["pred_logits"], captions, "")

        # visualize(out["pred_logits"][0], out["pred_boxes"][0], captions[0], batched_input[0]['image'], threshold=0.05)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, output_class, output_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(output_class[:-1], output_coord[:-1])
        ]

    def preprocess_image(self, batched_input):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_input]
        images = ImageList.from_tensors(images)

        return images

    def normalizer(self, x):
        pixel_mean = torch.Tensor(self.pixel_mean).to(x.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.pixel_std).to(x.device).view(3, 1, 1)
        return (x - pixel_mean) / pixel_std


def recover_to_cls_logits(logits, cate_to_token_mask_list, for_fill=float("-inf")):
    assert logits.shape[0] == len(cate_to_token_mask_list) # for batch align
    new_logits = torch.full(logits.shape, for_fill, device=logits.device)
    for bid, cate_to_token_mask in enumerate(cate_to_token_mask_list):
        for cate_cid in range(len(cate_to_token_mask)):
            logits_tmp = logits[bid, :, :cate_to_token_mask.shape[1]]
            logits_tmp = logits_tmp[:, cate_to_token_mask[cate_cid]]
            new_logits[bid, :, cate_cid] = torch.max(logits_tmp, dim=-1)[0]
    return new_logits


def build_groundingdino(args):
    model = safe_init(GroundingDINO, args)
    model = load_model(model, args.ckpt_path)

    return model