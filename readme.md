<p align="center">

  <h1 align="center">FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention</h1>
  <p align="center">
    <a href="https://wzpscott.github.io/">Zipeng Wang</a>
    ·
    <a href="https://www.danxurgb.net/">Dan Xu</a>
  </p>

  <h3 align="center"><a href="https://arxiv.org/pdf/2512.01540">Paper</a> | <a href="https://arxiv.org/abs/2512.01540">arXiv</a> | <a href="https://wzpscott.github.io/flashvggt_page/">Project Page</a>  | <a href="https://huggingface.co/papers/2512.01540">HuggingFace</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
TLDR: Accelerate VGGT with more efficient global attention for ~10x faster inference on 1K images and scaling to 3K+ images.
</p>
<br>

# Updates 
- [05/02/2026] Code and checkpoints for single-forward and streaming inference are released.

# Overview
<p align="center">
  <a href="">
    <img src="./assets/framework.png"Logo" width="95%">
  </a>
</p>
Instead of applying dense global attention across all tokens, FlashVGGT compresses spatial information from each frame into a compact set of descriptor tokens. Global attention is then computed as cross-attention between the full set of image tokens and this smaller descriptor set, significantly reducing computational overhead. Moreover, the compactness of the descriptors enables online inference over long sequences via a chunk-recursive mechanism that reuses cached descriptors from previous chunks. 

# Installation
## Environment Setup
First, you should clone the repository and create an anaconda environment.
```bash
git clone https://github.com/wzpscott/FlashVGGT.git
cd FlashVGGT
conda create -n flashvggt python=3.10 -y
conda activate flashvggt
```

Then,  You can use the following command to install the dependencies.
```bash
pip install -r requirements.txt
```

You can also install FlashVGGT as a package.
```bash
pip install -e . --no-deps
```

## Checkpoints
You can download the checkpoints for single-forward and streaming variants of FlashVGGT from the [HuggingFace](https://huggingface.co/ZipW/FlashVGGT). You should download the checkpoints to the `ckpts` folder.

```bash
# Create the checkpoints directory
mkdir -p ckpts

# Download the standard model
huggingface-cli download ZipW/FlashVGGT flashvggt.pt --local-dir ckpts

# Download the streaming model
huggingface-cli download ZipW/FlashVGGT flashvggt_stream.pt --local-dir ckpts
```

# Quick Start
We provide a demo script `demo_o3d.py` to visualize the 3D reconstruction results as point clouds using Open3D. The output is a `.ply` file that can be easily visualized with most 3D viewers.

### Usage Examples

**1. Standard FlashVGGT Inference:**
To run the standard FlashVGGT model on a folder of images:
```bash
python demo_o3d.py \
    --model FlashVGGT \
    --image_folder ./examples/garden/ \
    --output_dir outputs/
```

**2. Streaming FlashVGGT Inference:**
To run the streaming variant (FlashVGGTStream) which is optimized for longer sequences:
```bash
python demo_o3d.py \
    --model FlashVGGTStream \
    --image_folder ./examples/garden/ \
    --chunksize 10 \
    --output_dir outputs/
```

<details>
<summary><b>Key Arguments</b></summary>

- `--model`: Choose between `FlashVGGT` (single-forward) and `FlashVGGTStream` (streaming inference). Default is `FlashVGGT`.
- `--image_folder`: Path to the directory containing input images. Default is `./examples/garden/`.
- `--output_dir`: Directory where the generated `.ply` point cloud will be saved. Default is `outputs/`.
- `--chunksize`: Frame chunk size for `FlashVGGTStream` streaming inference. Default is `10`.
- `--max_points`: Maximum number of points to include in the output point cloud. Default is `1000000`.
- `--conf_threshold`: Percentage of low-confidence points to filter out (0-100). Default is `40.0`.
- `--kv_downfactor`: KV downfactor for attention compression. Default is `4`.
- `--keyframe_every`: Keyframe interval for the standard FlashVGGT model. Default is `200`.

</details>
