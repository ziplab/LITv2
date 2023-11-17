# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')

from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    args = parser.parse_args()
    return args

def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim


def li_sra_flops(h, w, dim):
    return 2 * h * w * 7 * 7 * dim

import math

def msa_flops(H, W, dim):
    N = H * W
    return N * dim * N * 2

def hilo_flops(H, W, l_dim, h_dim, sr_ratio):
    # H = int(N ** 0.5)
    # ws = sr_ratio = 4
    ws = sr_ratio
    Hp = ws * math.ceil(H / ws)
    Wp = ws * math.ceil(W / ws)
    Np = Hp * Wp

    nW = (Hp // ws) * (Wp // ws)
    window_len = ws * ws
    window_flops = window_len * window_len * h_dim * 2

    high_flops = nW * window_flops
    kv_len = (Hp // sr_ratio) * (Wp // sr_ratio)
    low_flops = Np * l_dim * kv_len * 2

    return high_flops + low_flops


def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)

    backbone = model.backbone
    _, H, W = input_shape
    l_dim = int(backbone.alpha * backbone.num_heads[2]) * 32
    h_dim = (backbone.num_heads[2] - int(backbone.alpha * backbone.num_heads[2])) * 32
    stage3 = hilo_flops(H // 16, W // 16, l_dim, h_dim, backbone.local_ws[2]) * len(backbone.layers[2].blocks)
    stage4 = msa_flops(H // 32, W // 32, backbone.num_heads[3] * 32) * len(backbone.layers[3].blocks)
    flops += stage3 + stage4
    return flops_to_string(flops), params_to_string(params)

def inference(args, logger):
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    _forward = model.forward

    flops, params = get_flops(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    inference(args, logger)



if __name__ == '__main__':
    main()