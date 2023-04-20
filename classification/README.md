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
torchrun --standalone --nnodes=1 --nproc-per-node=[num_gpus] main.py --cfg [path/to/config]
```

For example, you can train LITv2-S with 8 GPUs (1 node) by

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py --cfg configs/litv2-small.yaml
```

Note: In our experiments, we train all models ImageNet-1K with 8 GPUs under a total batch size of 1024.

## Evaluation

To evaluate a model, you can run

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=[num_gpus] main.py --cfg [path/to/config] --resume [path/to/checkpoint] --eval
```

For example, to evaluate LIT-S with 1 GPU, you can run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py --cfg configs/litv2-small.yaml --resume ./litv2_s.pth --eval
```

This should give

```
* Acc@1 82.044 Acc@5 95.666
Accuracy of the network on the 50000 test images: 82.0%
```

> Result could be slightly different based on you environment.

To test the throughput, you can run

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py --cfg configs/litv2-small.yaml --throughput
```


## ONNX and TensorRT Model Conversion
We suggest using Docker to convert models into ONNX and TensorRT as it eases the overhead of setting up environment.

First, pull the docker image from NVIDIA and create a container.
```bash
## pull docker image from NVIDIA
docker pull nvcr.io/nvidia/pytorch:23.03-py3

# run a container
docker run --gpus all -it --rm -v [path/to/lit_repo]:[working_dir_in_container] nvcr.io/nvidia/pytorch:23.03-py3

cd [working_dir_in_container]
pip install timm
pip install yacs
pip install termcolor
cd LITv2/classification/mm_modules/DCN
python setup.py install
```

Second, convert the model into ONNX. By default, we set the batch size into 64 and test with 224x224 images.

```bash
cd LITv2/classification
python convert_onnx.py --cfg configs/litv2-small.yaml --resume [path/to/pretrained_lit.pth]
```
The above command will create an onnx file at `./classifcation/{config.MODEL.NAME}.onnx`

Last, convert the onnx model into TensorRT and benchmark GPU throughput/latency.

```bash
# FP16 inference
trtexec --onnx=./litv2_small.onnx --saveEngine=litv2_small.trt --fp16


# Int8 inference
trtexec --onnx=./litv2_small.onnx --saveEngine=litv2_small.trt --int8
```

Example outputs on RTX 3090:

```bash
trtexec --onnx=./litv2_small.onnx --saveEngine=litv2_s.trt --fp16

...
...
[04/20/2023-10:23:30] [I] === Performance summary ===
[04/20/2023-10:23:30] [I] Throughput: 259.946 qps
[04/20/2023-10:23:30] [I] Latency: min = 3.79373 ms, max = 4.79907 ms, mean = 3.87182 ms, median = 3.85718 ms, percentile(90%) = 3.92688 ms, percentile(95%) = 3.94373 ms, percentile(99%) = 4.31226 ms
[04/20/2023-10:23:30] [I] Enqueue Time: min = 0.176758 ms, max = 0.518921 ms, mean = 0.358293 ms, median = 0.373535 ms, percentile(90%) = 0.432129 ms, percentile(95%) = 0.442139 ms, percentile(99%) = 0.498901 ms
[04/20/2023-10:23:30] [I] H2D Latency: min = 0.00195312 ms, max = 0.0563049 ms, mean = 0.00421258 ms, median = 0.00415039 ms, percentile(90%) = 0.00512695 ms, percentile(95%) = 0.00610352 ms, percentile(99%) = 0.00720215 ms
[04/20/2023-10:23:30] [I] GPU Compute Time: min = 3.76218 ms, max = 4.76569 ms, mean = 3.83965 ms, median = 3.82471 ms, percentile(90%) = 3.89636 ms, percentile(95%) = 3.91162 ms, percentile(99%) = 4.28033 ms
[04/20/2023-10:23:30] [I] D2H Latency: min = 0.0224609 ms, max = 0.0328369 ms, mean = 0.0279483 ms, median = 0.0279541 ms, percentile(90%) = 0.029541 ms, percentile(95%) = 0.0300293 ms, percentile(99%) = 0.0307617 ms
[04/20/2023-10:23:30] [I] Total Host Walltime: 3.01217 s
[04/20/2023-10:23:30] [I] Total GPU Compute Time: 3.00645 s
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
