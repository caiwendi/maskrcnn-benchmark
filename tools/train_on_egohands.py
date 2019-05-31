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

# for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
#     images = images.to(device)
#     targets = [target.to(device) for target in targets]
#     loss_dict = model(images, targets)
#
#     # losses = sum(loss for loss in loss_dict.values())
#     # print(losses)
#     exit()


"""

# def train(cfg, local_rank, distributed):
def train(cfg):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    # if distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[local_rank], output_device=local_rank,
    #         # this should be removed if we update BatchNorm stats
    #         broadcast_buffers=False,
    #     )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        # is_distributed=distributed,
        is_distributed=False,
        start_iter=arguments["iteration"],
    )

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

    return model


# def run_test(cfg, model, distributed):
def run_test(cfg, model):
    # if distributed:
    #     model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    # data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    # parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # parser.add_argument(
    #     "--config-file",
    #     default="",
    #     metavar="FILE",
    #     help="path to config file",
    #     type=str,
    # )
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument(
    #     "--skip-test",
    #     dest="skip_test",
    #     help="Do not test the final model",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    #
    # args = parser.parse_args()
    #
    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = num_gpus > 1
    #
    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://"
    #     )
    #     synchronize()

    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    # logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
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
    # logger.info("Running with config:\n{}".format(cfg))

    # model = train(cfg, args.local_rank, args.distributed)
    model = train(cfg)

    # if not args.skip_test:
    #     run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()


"""