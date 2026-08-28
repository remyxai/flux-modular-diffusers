---
library_name: diffusers
tags: [modular-diffusers, custom-block, flux, training-free]
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# <Method> for FLUX — <one-line> (Modular Diffusers custom block)

<intro: what it does, training-free/open-weight, arXiv link>.

![demo](assets/demo.png)

## Usage
```python
import torch
from diffusers import ModularPipeline
pipe = ModularPipeline.from_pretrained("remyxai/<name>-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16); pipe.to("cuda")
img = pipe(prompt="...").images[0]
```

## How it works
<mechanism + the injection pattern used>.

## Key parameters
| arg | default | meaning |
|---|---|---|

## Attribution & AI assistance
Port of <method> (<repo>, <license>). Authored with AI assistance (Claude), validated by Remyx AI. FLUX.1-dev (non-commercial).

## Citation
```bibtex
<verified via arxiv metadata>
```
