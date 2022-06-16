# Object Detection Code for LITv2


## Usage

1. Install [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)

2. Download COCO 2017 from the [official website](https://cocodataset.org/#download) and prepare the dataset. The directory structure should look like

   ```
   coco
   ├── annotations
   ├── train2017
   └── val2017
   ```

   Next, create a symbolic link to this repo by

   ```bash
   cd detection/
   mkdir data
   ln -s [path/to/coco] data/
   ```

3. Download LITv2 pretrained weights on ImageNet.



## Training

```bash
bash tools/dist_train.sh  <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```

For example, you can train LITv2-S with 8 GPUs by

```
bash tools/dist_train.sh configs/litv2/retinanet_litv2_s_fpn_1x_coco.py 8 --cfg-options model.pretrained=litv2_s.pth
```

## Inference

```bash
# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm

# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm
```

For example, you can test LITv2-S with RetinaNet on 8 GPUs by 

```bash
bash tools/dist_test.sh configs/litv2/retinanet_litv2_s_fpn_1x_coco.py retinanet_litv2_s_fpn_1x_coco.pth 8 --eval bbox
```


## Benchmark

To get the FLOPs, run

```bash
python tools/get_flops.py configs/litv2/retinanet_litv2_s_fpn_1x_coco.py
```

This should give

```bash
Input shape: (3, 1280, 800)
Flops: 242.39 GFLOPs
Params: 38.01 M
```

To test the FPS, run

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/benchmark.py \
       configs/litv2/retinanet_litv2_s_fpn_1x_coco.py \
       --checkpoint retinanet_litv2_s_fpn_1x_coco.pth \
       --launcher pytorch
```
