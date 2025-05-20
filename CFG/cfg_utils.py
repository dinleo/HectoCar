# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import wandb
from PIL import Image
import datetime
from omegaconf import OmegaConf
from detectron2.utils.events import EventWriter, get_event_storage

from detectron2.config import LazyConfig
from detectron2.engine import default_setup

from dotenv import load_dotenv
load_dotenv()

class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb file.
    """

    def __init__(self, runner_cfg, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        WDB = os.getenv('WANDB')
        if WDB:
            wandb.login(key=WDB)
        self._writer = wandb.init(
            entity=runner_cfg.wandb.entity,
            project=runner_cfg.wandb.project,
            name=runner_cfg.name,
            dir="wandb",
        )
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self._writer.log({k: v}, step=iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # visualize training samples
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                log_img = Image.fromarray(img.transpose(1, 2, 0))  # convert to (h, w, 3) PIL.Image
                log_img = wandb.Image(log_img, caption=img_name)
                self._writer.log({img_name: [log_img]})
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

    def close(self):
        if hasattr(self, "_writer"):
            self._writer.finish()

def try_get_key(cfg, *keys, default=None):
    """
    Try select keys from lazy cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default

def clean_files(dir_path):
    delete_files = {
        "coco_instances_results.json",
        "config.yaml.pkl",
        "instances_predictions.pth",
        "test_sub_coco_format.json",
        "test_sub_coco_format.json.lock",
    }

    for fname in os.listdir(dir_path):
        if fname in delete_files:
            full_path = os.path.join(dir_path, fname)
            if os.path.isfile(full_path):
                os.remove(full_path)

def get_config(config_path):
    cfg_file = os.path.join("CFG", config_path)
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in configs!".format(config_path))
    cfg = LazyConfig.load(cfg_file)
    return cfg

def default_setup_detectron2(cfg, args):
    # detectron2's default_setup uses hardcoded keys within a specific configuration namespace
    # e.g., cfg.train.seed, cfg.train.output_dir, cfg.train.cudnn_benchmark, cfg.train.float32_precision, and args.eval_only

    if cfg.runner.dev_test:
        date_str = datetime.datetime.now().strftime("%m%d_%H%M")
        cfg.runner.output_dir = os.path.join(cfg.runner.output_dir, date_str)
    elif not cfg.runner.eval_only and cfg.runner.wandb.entity and cfg.runner.wandb.project:
        cfg.runner.wandb_writer = WandbWriter(cfg.runner)

    cfg.train = cfg.runner
    args.eval_only = cfg.runner.eval_only
    cfg.solver.evaluator.output_dir = cfg.runner.output_dir

    default_setup(cfg, args)