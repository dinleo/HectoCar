#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import json
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter
)
from detectron2.checkpoint import DetectionCheckpointer

from models.model_utils import *
from solver.optimizer import ema
from CFG.cfg_utils import default_setup_detectron2, clean_files
from hf_up import upload

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings(action="ignore",
                        message="A parameter name that contains `beta` will be renamed internally to `bias`. Please use a different name to suppress this warning.")
warnings.filterwarnings(action="ignore",
                        message="A parameter name that contains `gamma` will be renamed internally to `weight`. Please use a different name to suppress this warning.")


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
            self,
            model,
            criterion,
            dataloader,
            optimizer,
            runner,
            grad_scaler=None,
            batch_size_scale=1,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        self.runner = runner
        self.amp = runner.amp.enabled
        self.iter_per_epoch = runner.iter_per_epoch
        self.max_epoch = runner.max_iter // runner.iter_per_epoch

        if self.amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

        # gradient clip hyper-params
        self.clip_grad_params = self.runner.clip_grad.params if self.runner.clip_grad.enabled else None

        # batch_size_scale
        self.batch_size_scale = batch_size_scale

        # criterion
        self.criterion = criterion

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        epoch = self.iter // self.iter_per_epoch
        progress = epoch / max((self.max_epoch - 1), 1)
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        inputs = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            outputs = self.model(inputs)
            if outputs is None:
                print(f"Skip {self.iter} due to Empty box")
                return None
            outputs["progress"] = progress
            loss_dict = self.criterion(outputs)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """

        if not losses.requires_grad or losses.grad_fn is None:
            print("[Warning] Loss tensor is not differentiable!")
            return None

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


def per_epoch(model):
    if hasattr(model, "print"):
        model.print()
    upload("")


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")
    test_loader = instantiate(cfg.dataloader.test)
    if 0 < cfg.runner.eval_sample:
        test_loader = instantiate(cfg.dataloader.test_sub)
        cfg.solver.evaluator.dataset_name = "test_sub"
    evaluator = instantiate(cfg.solver.evaluator)
    model.eval()
    if hasattr(model, "prepare"):
        model.prepare()

    if cfg.runner.model_ema.enabled:
        logger.info("Run evaluation with EMA.")
        with ema.apply_model_ema_and_restore(model):
            ret = evaluator(model, test_loader)
    else:
        ret = evaluator(model, test_loader)

    if eval_only:
        with PathManager.open(cfg.runner.output_dir + "/results.json", "w") as f:
            f.write(json.dumps(ret))
            f.flush()
        if not cfg.runner.dev_test:
            upload("")
        if hasattr(model, "print"):
            model.print()
    clean_files(cfg.runner.output_dir)
    return ret


def do_train(cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            solver.optimizer: instantiate to an optimizer
            solver.lr_scheduler: instantiate to a fvcore scheduler
            runner: other misc config defined in `CFG/{project_name}/runner_.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """

    if not torch.cuda.is_available():
        cfg.runner.device = "cpu"
        cfg.model.build.args.device = "cpu"
        print("CUDA is not available, fall back to CPU.")

    # instantiate model
    model = instantiate(cfg.model.build)
    model.to(cfg.runner.device)
    if hasattr(model, "prepare"):
        model.prepare()
        model.to(cfg.runner.device)

    # instantiate criterion
    criterion = instantiate(cfg.solver.criterion)
    criterion.to(cfg.runner.device)

    # build training loader
    train_loader = instantiate(cfg.dataloader.train)

    # create ddp model
    model = create_ddp_model(model, **cfg.runner.ddp)

    # build model ema
    ema.may_build_model_ema(cfg.runner, model)

    # instantiate optimizer
    cfg.solver.optimizer.params.model = model
    optim = instantiate(cfg.solver.optimizer)
    lr_scheduler = instantiate(cfg.solver.lr_scheduler)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        dataloader=train_loader,
        optimizer=optim,
        runner=cfg.runner
    )

    checkpointer = DetectionCheckpointer(
        model,
        f"{cfg.runner.output_dir}",
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg.runner, model)
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.runner.output_dir, cfg.runner.max_iter)
        output_dir = cfg.runner.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.runner.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if hasattr(cfg.runner, "wandb_writer"):
            writers.append(cfg.runner.wandb_writer)

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg.runner, model) if cfg.runner.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=lr_scheduler),
            # Log and Save
            hooks.PeriodicCheckpointer(checkpointer, **cfg.runner.checkpointer) if comm.is_main_process() else None,
            hooks.PeriodicWriter(writers, period=cfg.runner.log_period) if comm.is_main_process() else None,
            hooks.EvalHook(cfg.runner.iter_per_epoch, lambda: per_epoch(model)) if not cfg.runner.dev_test else None,
            # Eval
            hooks.EvalHook(cfg.runner.iter_per_epoch, lambda: do_test(cfg, model)) if cfg.runner.do_eval else None,
        ]
    )
    last_ckpt = cfg.runner.last_ckpt
    if cfg.runner.resume and last_ckpt != "":
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        checkpointer.load(last_ckpt)
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    check_frozen(model, max_depth=5)
    trainer.train(start_iter, cfg.runner.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup_detectron2(cfg, args)

    # Enable fast debugging by running several iterations to check for any bugs.
    if cfg.runner.dev_test:
        cfg.dataloader.train.batch_size = 4
        cfg.dataloader.train.num_workers = 1
        cfg.runner.iter_per_epoch = 50
        cfg.runner.epoch = 2
        cfg.runner.log_period = 1

    if cfg.runner.eval_only:
        model = instantiate(cfg.model.build)
        model.to(cfg.runner.device)
        model = create_ddp_model(model)

        # using ema for evaluation
        ema.may_build_model_ema(cfg.runner, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg.runner, model)).load(cfg.runner.last_ckpt)
        # Apply ema state for evaluation
        if cfg.runner.model_ema.enabled and cfg.runner.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        print(do_test(cfg, model, eval_only=True))
    else:
        do_train(cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    # parser.add_argument("--tsk-id", type=int, required=True, help="task id")
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
