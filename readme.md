<p align="center">

  <h1 align="center">FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=3w7X6NYAAAAJ">Zipeng Wang</a>
    ·
    <a href="https://www.danxurgb.net/">Dan Xu</a>
  </p>

  <h3 align="center"><a href="https://arxiv.org/pdf/2512.01540">Paper</a> | <a href="https://arxiv.org/abs/2512.01540">arXiv</a> | <a href="https://wzpscott.github.io/flashvggt_page/">Project Page</a>  | <a href="https://huggingface.co/papers/2512.01540">HuggingFace (coming soon)</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
TLDR: Accelerate VGGT with more efficient global attention for ~10x faster inference on 1K images and scaling to 3K+ images.
</p>
<br>


# Method
<p align="center">
  <a href="">
    <img src="./assets/framework.png"Logo" width="95%">
  </a>
</p>
Instead of applying dense global attention across all tokens, FlashVGGT compresses spatial information from each frame into a compact set of descriptor tokens. Global attention is then computed as cross-attention between the full set of image tokens and this smaller descriptor set, significantly reducing computational overhead. Moreover, the compactness of the descriptors enables online inference over long sequences via a chunk-recursive mechanism that reuses cached descriptors from previous chunks. 

# Code 
Coming soon. Please stay tuned.