# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import torch.utils.data
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.data.datasets.egohands import EgohandsDataset
from maskrcnn_benchmark.data.transforms.build import build_transforms
from maskrcnn_benchmark.data.build import make_batch_data_sampler
from maskrcnn_benchmark.data.collate_batch import BatchCollator, BBoxAugCollator
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def run_test():
    torch.cuda.empty_cache()
    iou_types = ("bbox",)



transforms = build_transforms(cfg)
dataset = EgohandsDataset([0, 1, 2], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], transforms)
num_iters = 70000
images_per_gpu = 1
start_iter = 0
sampler = torch.utils.data.sampler.RandomSampler(dataset)
batch_sampler = make_batch_data_sampler(dataset, sampler, False, images_per_gpu, num_iters, start_iter)
collator = BBoxAugCollator() if not True and cfg.TEST.BBOX_AUG.ENABLED else \
    BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
num_workers = cfg.DATALOADER.NUM_WORKERS
data_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=num_workers,
    batch_sampler=batch_sampler,
    collate_fn=collator,
)

model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

optimizer = make_optimizer(cfg, model)
scheduler = make_lr_scheduler(cfg, optimizer)

use_mixed_precision = cfg.DTYPE == "float16"
amp_opt_level = 'O1' if use_mixed_precision else 'O0'
model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

arguments = {}
arguments["iteration"] = 0

output_dir = "/home/lab/github/maskrcnn-benchmark/egohands_output"

logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
# logger.info("Using {} GPUs".format(num_gpus))
# logger.info(args)
#
# logger.info("Collecting env info (might take some time)")
# logger.info("\n" + collect_env_info())
#
# logger.info("Loaded configuration file {}".format(args.config_file))
# with open(args.config_file, "r") as cf:
#     config_str = "\n" + cf.read()
#     logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

save_to_disk = get_rank() == 0
checkpointer = DetectronCheckpointer(
    cfg, model, optimizer, scheduler, output_dir, save_to_disk
)
extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
arguments.update(extra_checkpoint_data)

checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )
