# Semantic Segmentation code for LITv2

## Installation

1. Install [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

2. Download ADE20K dataset from the [official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/). The directory structure should look like

   ```
   ade
   └── ADEChallengeData2016
       ├── annotations
       │   ├── training
       │   └── validation
       └── images
           ├── training
           └── validation
   ```

   Next, create a symbolic link to the dataset.

   ```bash
   cd segmentation/
   mkdir data
   ln -s [path/to/ade20k] data/
   ```

3. Download LITv2 pretrained weights on ImageNet.



## Training

To train a model with pre-trained weights, run:

```bash
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```

For example, to train a Semantic FPN model with a LITv2-S backbone on 8 GPUs, run:

```bash
tools/dist_train.sh configs/litv2/litv2_s_fpn_r50_512x512_80k_ade20k.py 8 --options model.pretrained=litv2_s.pth
```

## Inference

```bash
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU
```

For example, to evaluate a Semantic FPN model with a LITv2-S backbone, run:

```bash
tools/dist_test.sh configs/litv2/litv2_s_fpn_r50_512x512_80k_ade20k.py litv2_s_fpn_r50_512x512_80k_ade20k.pth 8 --eval mIoU
```



## Benchmark

To get the FLOPs, run

```bash
python tools/get_flops.py configs/litv2/litv2_s_fpn_r50_512x512_80k_ade20k.py
```

This should give

```bash
Input shape: (3, 512, 512)
Flops: 41.29 GFLOPs
Params: 31.45 M
```

To test the FPS, run

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/benchmark.py \
       configs/lit/retinanet_litv2_s_fpn_1x_coco.py \
       --checkpoint retinanet_litv2_s_fpn_1x_coco.pth \
       --launcher pytorch
```
