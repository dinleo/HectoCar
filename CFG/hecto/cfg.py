from CFG.cfg_utils import get_config
import os
data_dir = os.getenv("DATA", "inputs/data")
ckpt_dir = os.getenv("CKPT", "inputs/ckpt")
branch_name = os.getenv("BRANCH", "test")

# If set True, Enable fast debugging(batch=1, max_iter=200)
dev_test = False

# Base cfg
project_name = "hecto" # Cur Subdir
iter_per_epoch = 4000
epoch = 5
batch = 6
eval_sample = 1000 # If set -1, evaluate all testdata


# CFG Instance
dataloader = get_config(f"{project_name}/dataloader_.py").dataloader
model = get_config(f"{project_name}/models_.py").model
runner = get_config(f"{project_name}/runner_.py").runner
solver = get_config(f"{project_name}/solver_.py").solver


# modify dataloader
dataloader.train.num_workers = 8
dataloader.train.batch_size = batch
dataloader.train.image_root = f"{data_dir}/train"
dataloader.test.image_root = f"{data_dir}/test"


# modify model
# model.build.args.ckpt_path = f""
model.build.args.detr_backbone.args.ckpt_path = f"{ckpt_dir}/gdino.pth"


# modify solver
solver.optimizer.lr = 5e-4
solver.optimizer.weight_decay = 0.01

solver.lr_scheduler.epochs=epoch
solver.lr_scheduler.decay_epochs=epoch//2
solver.lr_scheduler.warmup_epochs = 1
solver.lr_scheduler.base_steps=iter_per_epoch


# modify runner
runner.name = f"{project_name}/{branch_name}" # Usage: output_dir_postfix, wandb_name
runner.device = "cuda"
runner.output_dir = f"./outputs/{runner.name}"
runner.dev_test = dev_test # only iterate 200 & no wandb logging
runner.max_iter = epoch * iter_per_epoch
runner.iter_per_epoch = iter_per_epoch

# Resume
runner.resume = False
runner.last_ckpt = ""

# eval
runner.do_eval = False
runner.eval_only = False
runner.eval_sample = eval_sample

# logging config
runner.checkpointer=dict(period=iter_per_epoch, max_to_keep=100)
runner.wandb.entity = "dinleo11"
runner.wandb.project = "Hecto"
