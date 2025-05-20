from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from datasets.gdino_loader import build_dataloader

dataloader = OmegaConf.create()
dataloader.train = L(build_dataloader)(
    image_root="",
    batch_size=5,
    num_workers=4,
    is_train=True,
)

dataloader.test = L(build_dataloader)(
    image_root="",
    batch_size=1,
    num_workers=4,
    is_train=False,
)

