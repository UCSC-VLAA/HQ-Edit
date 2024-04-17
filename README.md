# HQ-Edit: A High-Quality and High-Coverage Dataset for General Image Editing

[Dataset](https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit), [code](), and [model](https://huggingface.co/UCSC-VLAA/HQ-Edit) for [HQ-Edit](https://arxiv.org/abs/2404.09990).

A [working demo](https://huggingface.co/spaces/LAOS-Y/HQEdit) with our fine-tuned checkpoint is available.

Check [project website](https://thefllood.github.io/HQEdit_web/) for data examples and more.

![teaser image](figs/teaser.png)

## Dataset Summary
HQ-Edit is a high-quality and high-coverage instruction-based image editing dataset with around 200,000 edits collected with GPT-4V and DALL-E 3. HQ-Edit’s high-resolution images, rich in detail and accompanied by comprehensive editing prompts, substantially enhance the capabilities of existing image editing models.

## Create Your Own Dataset
Code Refactoring
## Quick Start
Make sure to install the libraries first:

```bash 
pip install accelerate transformers
pip install git+https://github.com/huggingface/diffusers
```

```python 
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

image_guidance_scale = 1.5
guidance_scale = 7.0
model_id = "MudeHui/HQ-Edit"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
resolution = 512
image = load_image(
    "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
).resize((resolution, resolution))

edit_instruction = "Turn sky into a cloudy one"
edited_image = pipe(
    prompt=edit_instruction,
    image=image,
    height=resolution,
    width=resolution,
    guidance_scale=image_guidance_scale,
    image_guidance_scale=image_guidance_scale,
    num_inference_steps=30,
).images[0]
edited_image.save("edited_image.png")
```

## Citation
If you find our HQ-Edit dataset or the fine-tuned checkpoint useful, please consider citing our paper:

```
@article{hui2024hq,
  title   = {HQ-Edit: A High-Quality Dataset for Instruction-based Image Editing},
  author  = {Hui, Mude and Yang, Siwei and Zhao, Bingchen and Shi, Yichun and Wang, Heng and Wang, Peng and Zhou, Yuyin and Xie, Cihang},
  journal = {arXiv preprint arXiv:2404.09990},
  year    = {2024}
}
```

