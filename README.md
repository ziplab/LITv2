# Fast Vision Transformers with HiLo Attention
Official PyTorch implementation of [Fast Vision Transformers with HiLo Attention](https://arxiv.org/abs/2205.13213).

By [Zizheng Pan](https://scholar.google.com.au/citations?user=w_VMopoAAAAJ&hl=en), [Jianfei Cai](https://scholar.google.com/citations?user=N6czCoUAAAAJ&hl=en), and [Bohan Zhuang](https://scholar.google.com.au/citations?user=DFuDBBwAAAAJ).

![hilo](.github/arch.png)


We introduce LITv2, a simple and effective ViT which performs favourably against the existing state-of-the-art methods across a spectrum of different model sizes with faster speed.

![hilo](.github/hilo.png)


The core of LITv2: **HiLo attention**. HiLo is inspired by the insight that high frequencies in an image capture local fine details and low frequencies focus on global structures, whereas a multi-head self-attention layer neglects the characteristic of different frequencies. Therefore, we propose to disentangle the high/low frequency patterns in an attention layer by separating the heads into two groups, where one group encodes high frequencies via self-attention within each local window, and another group performs the attention to model the global relationship between the average-pooled low-frequency keys from each window and each query position in the input feature map. 





## Usage

Code and pretrained weights will be released soon.



## Image Classification on ImageNet-1K

| Model   | Resolution | Params (M) | FLOPs (G) | Throughput (imgs/s) | Train Mem (GB) | Test Mem (GB) | Top-1 (%) |
| ------- | ---------- | ---------- | --------- | ------------------- | -------------- | ------------- | --------- |
| LITv2-S | 224        | 28         | 3.7       | 1,471               | 5.1            | 1.2           | 82.0      |
| LITv2-M | 224        | 49         | 7.5       | 812                 | 8.8            | 1.4           | 83.3      |
| LITv2-B | 224        | 87         | 13.2      | 602                 | 12.2           | 2.1           | 83.6      |
| LITv2-B | 384        | 87         | 39.7      | 198                 | 35.8           | 4.6           | 84.7      |

> Throughput and memory footprint are tested on one RTX 3090 based on a batch size of 64. Memory is measured by the peak memory usage with `torch.cuda.max_memory_allocated()`.

## Object Detection on COCO

### RetinaNet

| Backbone | Window Size | Params (M) | FLOPs (G) | FPS  | box AP |
| -------- | ----------- | ---------- | --------- | ---- | ------ |
| LITv2-S  | 2           | 38         | 242       | 18.7 | 44.0   |
| LITv2-S  | 4           | 38         | 230       | 20.4 | 43.7   |
| LITv2-M  | 2           | 59         | 348       | 12.2 | 46.0   |
| LITv2-M  | 4           | 59         | 312       | 14.8 | 45.8   |
| LITv2-B  | 2           | 97         | 481       | 9.5  | 46.7   |
| LITv2-B  | 4           | 97         | 430       | 11.8 | 46.3   |

### Mask R-CNN

| Backbone | Window Size | Params (M) | FLOPs (G) | FPS  | box AP | mask AP |
| -------- | ----------- | ---------- | --------- | ---- | ------ | ------- |
| LITv2-S  | 2           | 47         | 261       | 18.7 | 44.9   | 40.8    |
| LITv2-S  | 4           | 47         | 249       | 21.9 | 44.7   | 40.7    |
| LITv2-M  | 2           | 68         | 367       | 12.6 | 46.8   | 42.3    |
| LITv2-M  | 4           | 68         | 315       | 16.0 | 46.5   | 42.0    |
| LITv2-B  | 2           | 106        | 500       | 9.3  | 47.3   | 42.6    |
| LITv2-B  | 4           | 106        | 449       | 11.5 | 46.8   | 42.3    |



## Semantic Segmentation on ADE20K

### Semantic FPN

| Backbone | Params (M) | FLOPs (G) | FPS  | mIoU |
| -------- | ---------- | --------- | ---- | ---- |
| LITv2-S  | 31         | 41        | 42.6 | 44.3 |
| LITv2-M  | 52         | 63        | 28.5 | 45.7 |
| LITv2-B  | 90         | 93        | 27.5 | 47.2 |



## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/zip-group/LITv2/blob/main/LICENSE) file.



## Acknowledgement

This repository is built upon [DeiT](https://github.com/facebookresearch/deit), [Swin](https://github.com/microsoft/Swin-Transformer) and [LIT](https://github.com/zip-group/LIT), we thank the authors for their open-sourced code.

