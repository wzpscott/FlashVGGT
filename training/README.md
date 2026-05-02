# FlashVGGT Training

This directory contains the re-implementation of the training code for FlashVGGT, supporting both single-forward and streaming settings.

## 1. Installation

Install `flashvggt_training` as a package:

```bash
pip install -e .
```

## 2. Dataset Preparation

Download the following datasets and place them under the folder `FlashVGGT/data/training`.

The datasets used are:
- [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
- [MVSSynth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
- [ScanNet](http://www.scan-net.org/)
- [VKitti](https://europe.naverlabs.com/research/proxy-virtual-worlds/)
- [Mapillary](https://www.mapillary.com/dataset/depth)

### Data Processing

**For BlendedMVS, MVSSynth, and Mapillary:**
Prepare them following the instructions for WAI format data as described in the [MapAnything data processing guide](https://github.com/facebookresearch/map-anything/blob/main/data_processing/README.md).

**For ScanNet and VKitti:**
Download the datasets from the official repositories, [ScanNet](http://www.scan-net.org/), and [VKitti](https://europe.naverlabs.com/research/proxy-virtual-worlds/).

Once converted, you can test the correctness of the dataset processing by visualizing the point clouds:
```bash
python tools/test_dataset_pointcloud.py --dataset {dataset_name}
```

## 3. Training

### Single-Forward Training

To train the model in the single-forward setting, use the following command:

```bash
accelerate launch --num_processes {GPU_NUM} launch.py --config default exp_name=single_forward
```

### Streaming Setting

To train the model in the streaming setting, use the following command:

```bash
accelerate launch --num_processes 1 launch.py --config stream exp_name=stream
```

*Note: We use a causal mask for training the streaming model, which is more memory efficient.*
