import importlib
import os
import time
import random
from einops import rearrange
import numpy as np
import argparse
import shutil
import torch
import torch.autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn_limit, collation_fn_voxelmean, collation_fn_voxelmean_tta
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant

from util.nuscenes import nuScenes
from util.semantic_kitti import SemanticKITTI
from util.waymo import Waymo

from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv

from datetime import datetime

starttime = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

import os

import re

from operator import itemgetter
import debugpy
# torch.use_deterministic_algorithms(True)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_stratified_transformer.yaml', help='config file')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.debug = args.debug
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():

    starttime = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")    
    args = get_parser()

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    
    os.makedirs(f"./outputs/{starttime}_{args.arch}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # get model
    if args.arch == 'unet_spherical_transformer':
        from model.unet_spherical_transformer_SSC import Semantic as Model
        
        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(input_c=args.input_c, 
            m=args.m,
            classes=args.classes, 
            block_reps=args.block_reps, 
            block_residual=args.block_residual, 
            layers=args.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / args.quant_size_scale, 
            quant_size_sphere=window_size_sphere / args.quant_size_scale, 
            rel_query=args.rel_query, 
            rel_key=args.rel_key, 
            rel_value=args.rel_value, 
            drop_path_rate=args.drop_path_rate, 
            window_size_scale=args.window_size_scale, 
            grad_checkpoint_layers=args.grad_checkpoint_layers, 
            sphere_layers=args.sphere_layers,
            a=args.a,
        )

    else:
        
        module = importlib.import_module(f"model.{args.arch}")

        # from model.unet_spherical_transformer import Semantic_for_SSC as Model
        Model = module.Semantic_for_SSC
        
        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        segmentor = Model(input_c=args.input_c, 
            m=args.m,
            classes=args.classes, 
            block_reps=args.block_reps, 
            block_residual=args.block_residual, 
            layers=args.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / args.quant_size_scale, 
            quant_size_sphere=window_size_sphere / args.quant_size_scale, 
            rel_query=args.rel_query, 
            rel_key=args.rel_key, 
            rel_value=args.rel_value, 
            drop_path_rate=args.drop_path_rate, 
            window_size_scale=args.window_size_scale, 
            grad_checkpoint_layers=args.grad_checkpoint_layers, 
            sphere_layers=args.sphere_layers,
            a=args.a,
        )
        
        for param in segmentor.parameters():
            param.requires_grad = False
        
        # module = importlib.import_module(f"model.{args.arch}")
        model = module.SSC(
            input_c=args.input_c,
            segmentor=segmentor,
            m=args.m,
            classes=args.classes, 
            block_reps=args.block_reps, 
            block_residual=args.block_residual, 
            layers=args.layers_SSC if hasattr(args,"layers_SSC") else args.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / args.quant_size_scale, 
            quant_size_sphere=window_size_sphere / args.quant_size_scale, 
            rel_query=args.rel_query, 
            rel_key=args.rel_key, 
            rel_value=args.rel_value, 
            drop_path_rate=args.drop_path_rate, 
            window_size_scale=args.window_size_scale, 
            grad_checkpoint_layers=args.grad_checkpoint_layers, 
            sphere_layers=args.sphere_layers,
            a=args.a,
            **(args.SSC)
            )
        
        # if model.img_enc is not None:
        #     for param in model.img_enc.parameters():
        #         param.requires_grad = False


    
        # raise Exception('architecture {} not supported yet'.format(args.arch))
    
    # set optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" not in n and p.requires_grad],
            "lr": args.base_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" in n and p.requires_grad],
            "lr": args.base_lr * args.transformer_lr_scale,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if main_process():
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    # set loss func 
    class_weight = args.get("class_weight", None)
    class_weight = torch.tensor(class_weight).cuda() if class_weight is not None else None
    if main_process():
        logger.info("class_weight: {}".format(class_weight))
        logger.info("loss_name: {}".format(args.get("loss_name", "ce_loss")))
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=args.ignore_label, reduction='none' if args.loss_name == 'focal_loss' else 'mean').cuda()
    # criterion_occ = nn.CrossEntropyLoss(ignore_index=args.ignore_label, reduction='mean').cuda()
    criterion_occ = nn.BCEWithLogitsLoss(reduction='mean').cuda()

        
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            # TODO
            state_dict = {k.replace("module.", ""): v for k,v in checkpoint['state_dict'].items()}
            segmentor.load_state_dict(state_dict, strict=True)
            
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            if main_process():
                logger.info("=> loading checkpoint step 1")
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if main_process():
                logger.info("=> loading checkpoint step 2")
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loading checkpoint step 3")
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 'nuscenes':
        train_data = nuScenes(args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_train.pkl'], 
            voxel_size=args.voxel_size, 
            split='train',
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            ignore_label=args.ignore_label,
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    
    elif args.data_name == 'semantic_kitti':
        train_data = SemanticKITTI(args.data_root, 
            target=args.target,
            use_pseudo_voxels = args.get("use_pseudo_voxels", False),
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            label_mapping=args.label_mapping, 
            # ! rm todo fix augmentations
            rotate_aug=False, 
            flip_aug=False, 
            scale_aug=False, 
            scale_params=[0.95,1.05], 
            transform_aug=False, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    elif args.data_name == 'waymo':
        train_data = Waymo(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95, 1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    collate_fn = partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None)
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True, 
        collate_fn=collate_fn
    )

    val_transform = None
    args.use_tta = getattr(args, "use_tta", False)
    if args.data_name == 'nuscenes':
        val_data = nuScenes(data_path=args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_val.pkl'], 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None),
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    elif args.data_name == 'semantic_kitti':
        val_data = SemanticKITTI(data_path=args.data_root, 
            target=args.target,
            use_pseudo_voxels = args.get("use_pseudo_voxels", False),
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    elif args.data_name == 'waymo':
        val_data = Waymo(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.data_name == 'semantic_kitti':
        test_data = SemanticKITTI(data_path=args.data_root, 
            target=args.target,
            use_pseudo_voxels = args.get("use_pseudo_voxels", False),
            voxel_size=args.voxel_size, 
            split='test', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    else:
        test_data = None
        # raise ValueError("The dataset {} is not supported.".format(args.data_name))
    

    
    if main_process():
        logger.info("val_data samples: '{}'".format(len(val_data)))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
        if test_data is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)        
    else:
        print("NO_SAMPLER")
        val_sampler = None
        test_sampler = None
        
    if getattr(args, "use_tta", False):
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean_tta
        )
    else:
        print("NO_TTA")
        val_loader = torch.utils.data.DataLoader(val_data,
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean
        )
        
    if test_data is not None:
        test_loader = torch.utils.data.DataLoader(test_data,
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True, 
            sampler=test_sampler, 
            collate_fn=collation_fn_voxelmean
        )
    else:
        test_loader = None
        
    # set scheduler
    if args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        
    if args.test:
        print(len(test_data))
        print(len(test_loader))
        
        create_test_data(test_loader, model, criterion)
        exit()


    if args.val:
        if args.use_tta:
            validate_tta(val_loader, model, criterion)
        else:
            validate(val_loader, model, criterion)
            # validate_distance(val_loader, model, criterion)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))
            
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, (criterion, criterion_occ), optimizer, epoch, scaler, scheduler, gpu)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1
        
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            del loss_train 
            del mIoU_train
            del mAcc_train
            del allAcc_train
            torch.cuda.empty_cache()
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def focal_loss(output, target, class_weight, ignore_label, gamma, need_softmax=True, eps=1e-8):
    mask = (target != ignore_label)
    output_valid = output[mask]
    if need_softmax:
        output_valid = F.softmax(output_valid, -1)
    target_valid = target[mask]
    p_t = output_valid[torch.arange(output_valid.shape[0], device=target_valid.device), target_valid] #[N, ]
    class_weight_per_sample = class_weight[target_valid]
    focal_weight_per_sample = (1.0 - p_t) ** gamma
    loss = -(class_weight_per_sample * focal_weight_per_sample * torch.log(p_t + eps)).sum() / (class_weight_per_sample.sum() + eps)
    return loss


def train(train_loader, model, criteria, optimizer, epoch, scaler, scheduler, gpu):    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    loss_name = args.loss_name
    
    if loss_name == "ce_loss":
        criterion = criteria[0]
        criterion_occ = criteria[1]
    else: 
        criterion = criteria

    iou_class = 0
    accuracy_class = 0
    mIoU = 0


    for i, batch_data in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)

        data_time.update(time.time() - end)
        # torch.cuda.empty_cache()
        
        coord, xyz, feat, target, offset, file, image = batch_data
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
        for k,v in target.items():
            target[k] = v.cuda(non_blocking=True)
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target, offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)
        
        assert batch.shape[0] == feat.shape[0]
        
        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            
            outputs = model(sinput, xyz, batch, image)
            seg = outputs.pop("seg")

            

            targets = target
            total_loss = 0
            # assert set(target.keys()) <= set(outputs.keys())
            for key in outputs:
                output = outputs[key]
                factor = 1 if key == "1_1" else 0.1
                if "intermediate_preds" in key:
                    target = targets["1_1"]
                else:
                    # target = targets[key]
                    target = targets[key.split("@")[0]]
                    
                if target.shape[-1] == 1:
                    target = target[:, 0]  # for cls
                if loss_name == 'focal_loss':
                    loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
                elif "occ" in key:
                    loss = criterion_occ(output[target!=255], target[target!=255].float())
                elif loss_name == 'ce_loss':
                    
                    loss = criterion(output, target) * factor# + criterion_occ(occ_grid, occ_target)
                    # loss =criterion_occ(occ_grid, occ_target)
                else:
                    raise ValueError("such loss {} not implemented".format(loss_name))
                total_loss += loss
                
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()
        
        output = outputs["1_1"]
        target = targets["1_1"]
        
        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        occupancy_intersection,occupancy_union, intersection, union, target = intersectionAndUnionGPU(output, target, args.classes + 1, args.ignore_label)
        intersection, union, target = intersection[1:], union[1:], target[1:]
        if args.multiprocessing_distributed:
            dist.all_reduce(occupancy_intersection),dist.all_reduce(occupancy_union), dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        occupancy_intersection,occupancy_union, intersection, union, target = occupancy_intersection.cpu().numpy(),occupancy_union.cpu().numpy(),intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        iou = occupancy_intersection / occupancy_union
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)


        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '
                        'IoU {iou:.4f} '
                        'mIoU {mIoU:.4f} '
                        'Accuracy {accuracy:.4f}.'
                        'best mIoU {best_iou:.4f}'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time, data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter,
                                                        lr=lr,
                                                        iou = iou,
                                                        mIoU = mIoU,                                                        
                                                        accuracy=accuracy,
                                                        best_iou = best_iou))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc

def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    inference_time = AverageMeter()


    loss_name = args.loss_name

    model.eval()
    end = time.time()
    
    iou_class = 0
    accuracy_class = 0
    mIoU = 0

    
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        (coord, xyz, feat, target, offset, inds_reconstruct, file, image) = batch_data
        prefix = re.sub("\D","",file[0][-12:])
        inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)

        for k,v in target.items():
            target[k] = v.cuda(non_blocking=True)    
        coord, xyz, feat, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        targets = target
        target=targets["1_1"]
        with torch.no_grad():
            before_inference = time.time()
            outputs = model(sinput, xyz, batch, image) 
            inference_time.update(time.time() - before_inference)
            seg = outputs.pop("seg")
            output = outputs["1_1"]

        
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]

        
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        # intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        _,_, intersection, union, target = intersectionAndUnionGPU(output, target, args.classes + 1, args.ignore_label)
        intersection, union, target = intersection[1:], union[1:], target[1:]
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()


        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Inference {inference_time.val:.3f} ({inference_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'mIoU {mIoU:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          inference_time=inference_time,
                                                          loss_meter=loss_meter,
                                                          mIoU = mIoU,
                                                          accuracy=accuracy))

    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc

def validate_tta(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()


    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data_list in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        with torch.no_grad():
            output = 0.0
            for batch_data in batch_data_list:

                (coord, xyz, feat, target, offset, inds_reconstruct,file,image) = batch_data
                inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

                offset_ = offset.clone()
                offset_[1:] = offset_[1:] - offset_[:-1]
                batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

                coord = torch.cat([batch.unsqueeze(-1), coord], -1)
                spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
                coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
                batch = batch.cuda(non_blocking=True)

                sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

                assert batch.shape[0] == feat.shape[0]
                
                output_i = model(sinput, xyz, batch)
                output_i = F.softmax(output_i[inds_reconstruct, :], -1)
                
                output = output + output_i
            output = output / len(batch_data_list)
            
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes + 1, args.ignore_label)
        intersection, union, target = intersection[1:], union[1:], target[1:]
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate_distance(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # For validation on points with different distance
    intersection_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    union_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    target_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]


    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        (coord, xyz, feat, target, offset, inds_reverse, file, image) = batch_data
        prefix = re.sub("\D","",file[0][-12:])
        
        
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
        for k,v in target.items():
            target[k] = v.cuda(non_blocking=True)
        coord, xyz, feat, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]

        targets = target

        with torch.no_grad():
            outputs = model(sinput, xyz, batch, model)
            # todo
            # output = output[inds_reverse, :]
            
            for key in targets:
                if "occ" in key:
                    continue
                target = targets[key]
                output = outputs[key]

                if loss_name == 'focal_loss':
                    loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
                elif "occ" in key:
                    continue
                    # loss = criterion_occ(output[target!=255], target[target!=255].float())
                elif loss_name == 'ce_loss':
                    loss = criterion(output, target)
                else:
                    raise ValueError("such loss {} not implemented".format(loss_name))

        output = outputs["1_1"]
        target = targets["1_1"]
        


        output = output.max(1)[1] # todo same as above
        
        LABEL = output
        LABEL[LABEL == 255] = 0
        print(LABEL.shape)
        print(f"{prefix}.label")
        remapped_output = np.moveaxis(LABEL.cpu().numpy(), [0, 1, 2,3], [0, 1, 3, 2]).reshape(-1).astype(np.uint16)
        remapped_output = val_loader.dataset.get_inv_remap_lut()[remapped_output].astype(np.uint16)
        # TODO change hardcoded sequence
        os.makedirs("./preds/sequences/08/predictions/",exist_ok=True)
        remapped_output.tofile(f"./preds/sequences/08/predictions/{prefix}.label")
        # TODO -------------


        
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        
        occupancy_intersection,occupancy_union, intersection, union, target = intersectionAndUnionGPU(output, target, args.classes + 1, args.ignore_label)
        intersection, union, target = intersection[1:], union[1:], target[1:]
        
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():

        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes+1):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


def create_test_data(test_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(test_loader):

        data_time.update(time.time() - end)
    
        (coord, xyz, feat, _, offset, inds_reverse, file, image) = batch_data
        prefix = re.sub("\D","",file[0][-12:])
        sequence_number = re.sub("\D","",file[0][:-12])
        print(sequence_number)
        

        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
        coord, xyz, feat, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]


        with torch.no_grad():
            outputs = model(sinput, xyz, batch, image)
            
        output = outputs["1_1"]

        output = output.max(1)[1] # todo same as above
        
        LABEL = output
        LABEL[LABEL == 255] = 0
        print(LABEL.shape)
        print(f"{prefix}.label")
        remapped_output = np.moveaxis(LABEL.cpu().numpy(), [0, 1, 2,3], [0, 1, 3, 2]).reshape(-1).astype(np.uint16)
        remapped_output = test_loader.dataset.get_inv_remap_lut()[remapped_output].astype(np.uint16)
        # TODO change hardcoded sequence
        os.makedirs(f"./preds/sequences/{sequence_number}/predictions/",exist_ok=True)
        remapped_output.tofile(f"./preds/sequences/{sequence_number}/predictions/{prefix}.label")
        # TODO -------------

        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        .format(i + 1, len(test_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          ))

    dist.barrier()

# TODO midvoxio is in cuda2 / thesis and has to be installed from there
# import midvoxio
# import midvoxio.voxio as voxio

def get_cmap_semanticKITTI20():
  colors = np.array([
    [0  , 0  , 0, 255],
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255]]).astype(np.uint8)

  return colors

cmap = get_cmap_semanticKITTI20()
# ignore_color = np.array([128, 128,128,128]).reshape(1,4)
# cmap = np.concatenate((cmap,ignore_color),axis=0)
# cmap[1] = ignore_color
cmap = np.concatenate((cmap,np.zeros((256-20,4))),axis=0)

def grid_to_vox(occ, filename, binary = False):
    indices=None
    if binary:
        vox = np.ones((256,32,256,4), dtype=np.int32) * occ.squeeze(0)[:,:,:,None]
        palette = np.ones((256,4), dtype=np.int32)* 255
    else:
        vox = (cmap[occ]).astype(np.int32)
        vox = vox[:,:,:,[2,1,0,3]] / 255
        # palette = cmap[:,[2,1,0,3]].astype(int)
        palette = cmap.astype(int)
        occ = occ.copy()
        # occ[occ==255] = 21
        occ[occ==255] = 0
        occ = occ + 1
        occ = occ.transpose(2,0,1)
        al,bl,cl = occ.shape
        a,b,c = np.linspace(0,al-1,al),np.linspace(0,bl-1,bl),np.linspace(0,cl-1,cl)
        grid = np.stack(np.meshgrid(a,b,c,indexing="ij"),axis=-1)
        out = np.concatenate((grid,occ[:,:,:,None]),axis=-1).astype(int).reshape(-1,4).tolist()
        indices = out
        indices = list(map(tuple, indices))
        indices = list(map(lambda x: (*x[:3], False) if x[3] == 1 else x, indices))
        
    vox = vox.transpose(2,0,1,3)
    # voxio.write_list_to_vox(vox,filename,palette_arr=palette,indices=indices)
    # grid_to_vox(label.astype(np.int32)[:256,:,:256])




def to_vox(segmentation, output, label,i, binary=False):
    
    # draw segmentation 
    draw_segmentation = segmentation[0]
    draw_segmentation = torch.cat((torch.ones(1,256,256,32).cuda()*1e-10,draw_segmentation), 0)
    draw_segmentation = draw_segmentation.max(0)[1]
    draw_segmentation = rearrange(draw_segmentation, "d w h -> d h w")
    draw_segmentation = draw_segmentation.cpu().detach().numpy()
    grid_to_vox(draw_segmentation, f"outputs/{starttime}_{args.arch}/pc_segmentation_{i}.vox", binary=binary)

    
    # --------------------
    draw_output = output[0,:20]
    # draw_output = torch.cat((torch.ones(1,256,256,32).cuda()*1e-10,draw_output), 0)
    draw_output = draw_output.max(0)[1]
    # draw_output[draw_output != 0] = 1
    draw_output = rearrange(draw_output, "d w h -> d h w")
    draw_output = draw_output.cpu().detach().numpy()
    print(np.unique(draw_output))
    grid_to_vox(draw_output, f"outputs/{starttime}_{args.arch}/prediction_{i}.vox", binary=binary)
    
    grid_to_vox(label[0].cpu().numpy(), f"outputs/{starttime}_{args.arch}/label_{i}.vox")

if __name__ == '__main__':
    import gc
    gc.collect()
    # try:
    main()
    # except Exception as ex:
    #     print(traceback.format_exc())