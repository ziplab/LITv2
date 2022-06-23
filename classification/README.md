# Classification Code for LITv2

## Dataset Preparation

Download the ImageNet 2012 dataset from [here](http://image-net.org/), and prepare the dataset based on this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4). The file structure should look like:

```
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```



## Training

First, activate your python environment

```bash
conda activate lit
```

Make sure you have the correct ImageNet `DATA_PATH` in `config/*.yaml`. 

To train a model on ImageNet:

```bash
python -m torch.distributed.launch --nproc_per_node [num_gpus] --master_port 13335  main.py --cfg [path/to/config]
```

For example, you can train LITv2-S with 8 GPUs by

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 13335  main.py --cfg configs/litv2-small.yaml
```

Note: In our experiments, we train all models ImageNet-1K with 8 GPUs under a total batch size of 1024.

## Evaluation

To evaluate a model, you can run

```bash
python -m torch.distributed.launch --nproc_per_node [num_gpus] --master_port 13335  main.py --cfg [path/to/config] --eval
```

For example, to evaluate LIT-S with 1 GPU, you can run:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 13335  main.py --cfg configs/litv2-small.yaml --eval
```

This should give

```
* Acc@1 82.044 Acc@5 95.666
Accuracy of the network on the 50000 test images: 82.0%
```

> Result could be slightly different based on you environment.

To test the throughput, you can run

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 13335  main.py --cfg configs/litv2-small.yaml --throughput
```

## Results


| Model   | Resolution | Params (M) | FLOPs (G) | Throughput (imgs/s) | Train Mem (GB) | Test Mem (GB) | Top-1 (%) | Download                                                     |
| ------- | ---------- | ---------- | --------- | ------------------- | -------------- | ------------- | --------- | ------------------------------------------------------------ |
| LITv2-S | 224        | 28         | 3.7       | 1,471               | 5.1            | 1.2           | 82.0      | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_s.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_s_log.txt) |
| LITv2-M | 224        | 49         | 7.5       | 812                 | 8.8            | 1.4           | 83.3      | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_m.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_m_log.txt) |
| LITv2-B | 224        | 87         | 13.2      | 602                 | 12.2           | 2.1           | 83.6      | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_b.pth) & [log](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_b_log.txt) |
| LITv2-B | 384        | 87         | 39.7      | 198                 | 35.8           | 4.6           | 84.7      | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_b_384.pth) |



## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ziplab/LITv2/blob/main/LICENSE) file.
