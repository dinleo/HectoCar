from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

from solver.criterion.hecto_criterion import HectoCriterion
from solver.optimizer.scheduler import modified_coco_scheduler
from solver.evaluator.coco_eval import CocoDefaultEvaluator
import torch

solver = OmegaConf.create()
solver.optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)

solver.criterion = L(HectoCriterion)()

solver.lr_scheduler = L(modified_coco_scheduler)(
    epochs=10,
    decay_epochs=5,
    warmup_epochs=1,
    base_steps=1000,
)

solver.evaluator = L(CocoDefaultEvaluator)(
    dataset_name="test",
)