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
bash tools/dist_train.sh  <CONFIG_FILE> <GPU_NUM> 
```

For example, you can train LITv2-S with 1 GPU by

```
bash tools/dist_train.sh configs/litv2/retinanet_litv2_s_fpn_1x_coco.py 1
```

## Inference

```bash
# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> 

# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE>
```

For example, you can test LITv2-S with RetinaNet on 8 GPUs by 

```bash
bash tools/dist_test.sh configs/litv2/retinanet_litv2_s_fpn_1x_coco.py retinanet_litv2_s_fpn_1x_coco.pth 8
```



## Benchmark

To get the FLOPs, run

```bash
python tools/analysis_tools/get_flops.py configs/litv2/retinanet_litv2_s_fpn_1x_coco.py
```

This should give

```bash
Input shape: (3, 1280, 800)
Flops: 242.39 GFLOPs
Params: 38.01 M
```





To test the FPS, run

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs/litv2/retinanet_litv2_s_fpn_1x_coco.py \
       --launcher pytorch
```

## Results

#### RetinaNet

| Backbone | Window Size | Params (M) | FLOPs (G) | FPS  | box AP | Config                                                       | Download                                                     |
| -------- | ----------- | ---------- | --------- | ---- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LITv2-S  | 2           | 38         | 242       | 18.7 | 44.0   | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/retinanet_litv2_s_fpn_1x_coco.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_s_fpn_1x_coco.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_s_fpn_1x_coco_log.json) |
| LITv2-S  | 4           | 38         | 230       | 20.4 | 43.7   | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/retinanet_litv2_s_fpn_1x_coco_ws_4.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_s_fpn_1x_coco_ws_4.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_s_fpn_1x_coco_ws_4_log.json) |
| LITv2-M  | 2           | 59         | 348       | 12.2 | 46.0   | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/retinanet_litv2_m_fpn_1x_coco.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_m_fpn_1x_coco.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_m_fpn_1x_coco_log.json) |
| LITv2-M  | 4           | 59         | 312       | 14.8 | 45.8   | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/retinanet_litv2_m_fpn_1x_coco_ws_4.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_m_fpn_1x_coco_ws_4.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_m_fpn_1x_coco_ws_4_log.json) |
| LITv2-B  | 2           | 97         | 481       | 9.5  | 46.7   | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/retinanet_litv2_b_fpn_1x_coco.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_b_fpn_1x_coco.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_b_fpn_1x_coco_log.json) |
| LITv2-B  | 4           | 97         | 430       | 11.8 | 46.3   | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/retinanet_litv2_b_fpn_1x_coco_ws_4.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_b_fpn_1x_coco_ws_4.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/retinanet_litv2_b_fpn_1x_coco_ws_4_log.json) |

#### Mask R-CNN

| Backbone | Window Size | Params (M) | FLOPs (G) | FPS  | box AP | mask AP | Config                                                       | Download                                                     |
| -------- | ----------- | ---------- | --------- | ---- | ------ | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LITv2-S  | 2           | 47         | 261       | 18.7 | 44.9   | 40.8    | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/mask_rcnn_litv2_s_fpn_1x_coco.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_s_fpn_1x_coco.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_s_fpn_1x_coco_log.json) |
| LITv2-S  | 4           | 47         | 249       | 21.9 | 44.7   | 40.7    | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/mask_rcnn_litv2_s_fpn_1x_coco_ws_4.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_s_fpn_1x_coco_ws_4.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_s_fpn_1x_coco_ws_4_log.json) |
| LITv2-M  | 2           | 68         | 367       | 12.6 | 46.8   | 42.3    | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/mask_rcnn_litv2_m_fpn_1x_coco.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_m_fpn_1x_coco.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_m_fpn_1x_coco_log.json) |
| LITv2-M  | 4           | 68         | 315       | 16.0 | 46.5   | 42.0    | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/mask_rcnn_litv2_m_fpn_1x_coco_ws_4.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_m_fpn_1x_coco_ws_4.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_m_fpn_1x_coco_ws_4_log.json) |
| LITv2-B  | 2           | 106        | 500       | 9.3  | 47.3   | 42.6    | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/mask_rcnn_litv2_b_fpn_1x_coco.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_b_fpn_1x_coco.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_b_fpn_1x_coco_log.json) |
| LITv2-B  | 4           | 106        | 449       | 11.5 | 46.8   | 42.3    | [config](https://github.com/ziplab/LITv2/blob/main/detection/configs/litv2/mask_rcnn_litv2_b_fpn_1x_coco_ws_4.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_b_fpn_1x_coco_ws_4.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/mask_rcnn_litv2_b_fpn_1x_coco_ws_4_log.json) |

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ziplab/LITv2/blob/main/LICENSE) file.
