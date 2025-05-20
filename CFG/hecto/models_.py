from omegaconf import OmegaConf
from detectron2.config import LazyCall as L

from models.hecto import build_hecto
from models.groundingdino import build_groundingdino, build_backbone, build_transformer

model = OmegaConf.create()
model.modelname = "hecto"

backbone_args = dict(
    backbone="swin_B_384_22k",
    return_interm_indices=[1, 2, 3],
    backbone_freeze_keywords=None,
    hidden_dim=256,
    position_embedding="sine",
    pe_temperatureH=20,
    pe_temperatureW=20,
    use_checkpoint="True",
)
transformer_args = dict(
    d_model=256,
    dropout=0.0,
    nhead=8,
    num_queries=900,
    dim_feedforward=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    normalize_before=False,
    return_intermediate_dec=True,
    query_dim=4,
    activation="relu",
    num_patterns=0,
    num_feature_levels=4,
    enc_n_points=4,
    dec_n_points=4,
    learnable_tgt_init=True,
    two_stage_type= "standard",
    embed_init_tgt=True,
    use_text_enhancer=True,
    use_fusion_layer=True,
    use_checkpoint=True,
    use_transformer_ckpt=True,
    use_text_cross_attention=True,
    text_dropout=0.0,
    fusion_dropout=0.0,
    fusion_droppath=0.1,
)
gdino_args = dict(
    num_queries=900,
    aux_loss=True,
    iter_update=True,
    query_dim=4,
    num_feature_levels=4,
    nheads=8,
    dec_pred_bbox_embed_share=True,
    two_stage_type="standard",
    two_stage_bbox_embed_share=False,
    two_stage_class_embed_share=False,
    num_patterns=0,
    dn_number=0,
    dn_box_noise_scale=1.0,
    dn_label_noise_ratio=0.5,
    dn_labelbook_size=2000,
    text_encoder_type="bert-base-uncased",
    sub_sentence_present=True,
    max_text_len=256,
    device="cuda",
)

model.\
    build = L(build_hecto)(
        args=dict(
            ckpt_path="",
            detr_backbone = L(build_groundingdino)(
                args=dict(
                    ckpt_path="",
                    backbone=L(build_backbone)(
                        args=backbone_args
                    ),
                    transformer=L(build_transformer)(
                        args=transformer_args
                    ),
                    mode="stage1",
                    **gdino_args
            ))
    ))
