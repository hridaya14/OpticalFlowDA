from __future__ import print_function, division
import argparse
import numpy as np
from pathlib import Path

from model.core.ur2p_inspired.model import FogFormerEnhancer
import torch
import torch.nn as nn
import core.datasets_video as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
import random
import os
import torch.distributed as dist
import torch.multiprocessing as mp




try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    def train_memflow(self, gpu, cfg):
        rank = cfg.node_rank * cfg.gpus + gpu
        torch.cuda.set_device(rank)

        if cfg.DDP:
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=cfg.world_size,
                                    rank=rank,
                                    group_name='mtorch')
            model = nn.SyncBatchNorm.convert_sync_batchnorm(build_network(cfg)).cuda()
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        loss_func = sequence_loss

        if 'freeze_encoder' in cfg and cfg.freeze_encoder:
            print("[Freeze feature, context  and qk encoder]")
            for param in model.module.cnet.parameters():
                param.requires_grad = False
            for param in model.module.fnet.parameters():
                param.requires_grad = False
            for param in model.module.att.parameters():
                param.requires_grad = False

        if rank == 0:
            loguru_logger.info("Parameter Count: %d" % count_parameters(model))

        if cfg.restore_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
            ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
            ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
            if 'module' in list(ckpt_model.keys())[0]:
                model.load_state_dict(ckpt_model, strict=True)
            else:
                model.module.load_state_dict(ckpt_model, strict=True)

        model.train()

        # if cfg.eval_only:
        #     for val_dataset in cfg.validation:
        #         results = {}
        #         if val_dataset == 'sintel_train':
        #             results.update(evaluate_MemFlowNet.validate_sintel(model.module, cfg, rank))
        #         elif val_dataset == 'spring_train':
        #             results.update(evaluate_MemFlowNet.validate_spring(model.module, cfg, rank))
        #         elif val_dataset == 'spring_subset_val':
        #             results.update(
        #                 evaluate_MemFlowNet.validate_spring(model.module, cfg, rank, split='subset_val'))
        #         elif val_dataset == 'things':
        #             results.update(evaluate_MemFlowNet.validate_things(model.module, cfg, rank))
        #         elif val_dataset == 'kitti':
        #             results.update(evaluate_MemFlowNet.validate_kitti(model.module, cfg, rank))
        #         elif val_dataset == 'sintel_submission':
        #             evaluate_MemFlowNet.create_sintel_submission(model.module, cfg, output_path=cfg.suffix)
        #         elif val_dataset == 'spring_submission':
        #             evaluate_MemFlowNet.create_spring_submission(model.module, cfg, output_path=cfg.suffix, rank=rank)
        #         elif val_dataset == 'kitti_submission':
        #             evaluate_MemFlowNet.create_kitti_submission(model.module, cfg, output_path=cfg.suffix)
        #         print(results)
        #     return

        if cfg.DDP:
            train_sampler, train_loader = datasets.fetch_dataloader(cfg, DDP=cfg.DDP, rank=rank)
        else:
            train_loader = datasets.fetch_dataloader(cfg, DDP=cfg.DDP, rank=rank)

        optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        logger = Logger(model, scheduler, cfg)

        epoch = 0
        if cfg.restore_steps > 1:
            print("[Loading optimizer from {}]".format(cfg.restore_ckpt))
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.total_steps = cfg.restore_steps - 1
            total_steps = cfg.restore_steps
            epoch = ckpt['epoch']
            for _ in range(total_steps):
                scheduler.step()

        should_keep_training = True
        while should_keep_training:

            epoch += 1
            if cfg.DDP:
                train_sampler.set_epoch(epoch)

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                images, flows, valids = [x.cuda() for x in data_blob]
                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    images = (images + stdv * torch.randn(*images.shape).cuda()).clamp(0.0, 255.0)

                output = {}
                # flow prediction
                images = 2 * (images / 255.0) - 1.0
                b = images.shape[0]
                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision, dtype=torch.bfloat16):
                    # B*C*N-1*H*W,                    B*N-1*C*H*W
                    query, key, net, inp = model.module.encode_context(images[:, :-1, ...])

                    coords0, coords1, fmaps = model.module.encode_features(images)
                    values = None
                    video_flow_predictions = []  # frame by frame
                    for ti in range(0, cfg.input_frames - 1):
                        if ti < cfg.num_ref_frames:
                            ref_values = values
                            ref_keys = key[:, :, :ti + 1]
                        else:
                            indices = [torch.randperm(ti)[:cfg.num_ref_frames - 1] for _ in range(b)]
                            ref_values = torch.stack([
                                values[bi, :, indices[bi]] for bi in range(b)
                            ], 0)
                            ref_keys = torch.stack([
                                key[bi, :, indices[bi]] for bi in range(b)
                            ], 0)
                            ref_keys = torch.cat([ref_keys, key[:, :, ti].unsqueeze(2)], dim=2)

                        # predict flow from frame ti to frame ti+1
                        flow_pr, current_value = model.module.predict_flow(net[:, ti], inp[:, ti], coords0, coords1,
                                                                    fmaps[:, ti:ti + 2], query[:, :, ti], ref_keys,
                                                                        ref_values)
                        values = current_value if values is None else torch.cat([values, current_value], dim=2)
                        video_flow_predictions.append(torch.stack(flow_pr, dim=0))
                    # loss function
                    video_flow_predictions = torch.stack(video_flow_predictions, dim=2)  # Iter, B, N-1, 2, H, W

                    loss, metrics, _ = loss_func(video_flow_predictions, flows, valids, cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                metrics.update(output)
                metrics['scale'] = scaler.get_scale()
                if rank == 0:
                    logger.push(metrics)

                # if total_steps % cfg.val_freq == cfg.val_freq - 1 and rank == 0:
                #     print('start validation')
                #     PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1, cfg.name)
                #     torch.save({
                #         'iteration': total_steps,
                #         'epoch': epoch,
                #         'optimizer': optimizer.state_dict(),
                #         'model': model.module.state_dict(),
                #     }, PATH)
                # if total_steps % cfg.val_freq == cfg.val_freq - 1:
                #     results = {}
                #     for val_dataset in cfg.validation:
                #         if val_dataset == 'sintel_train':
                #             results.update(evaluate_MemFlowNet.validate_sintel(model.module, cfg, rank))
                #         elif val_dataset == 'kitti':
                #             results.update(evaluate_MemFlowNet.validate_kitti(model.module, cfg, rank))
                #         elif val_dataset == 'spring_subset_val':
                #             results.update(evaluate_MemFlowNet.validate_spring(model.module, cfg, rank, split='subset_val'))

                    model.train()
                if total_steps % cfg.val_freq == cfg.val_freq - 1 and rank == 0:
                    logger.write_dict(results)

                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    dist.barrier()

                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        if rank == 0:
            PATH = cfg.log_dir + f'/{cfg.name}.pth'
            torch.save(model.module.state_dict(), PATH)
        self.cleanup()
        return


    def train_enhancement_module(self,gpu, cfg):

        rank = cfg.node_rank * cfg.gpus + gpu
        torch.cuda.set_device(rank)

        if cfg.DDP:
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=cfg.world_size,
                                    rank=rank,
                                    group_name='mtorch')
            model = nn.SyncBatchNorm.convert_sync_batchnorm(build_network(cfg)).cuda()
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])


        if rank == 0:
            loguru_logger.info("Parameter Count: %d" % count_parameters(model))

        # Build enhancement module
        enhance_model = FogFormerEnhancer()






