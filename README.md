# Flux WebUI

Minimal Flux Web UI powered by Gradio & Diffusers.

Supports:

- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell): A fast model with 4 step inference.
- [FLUX.1-merged](https://huggingface.co/sayakpaul/FLUX.1-merged): Flux.1-dev with just 8 steps!

![ui.png](ui.png)

# Install

## 1. One click install

The easiest way is to use https://pinokio.computer

## 2. Install manually

First install torch:

```
# Windows/Linux CUDA
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 xformers --index-url https://download.pytorch.org/whl/cu121

# Mac MPS
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

Next, install the app dependencies:

```
pip install -r requirements.txt
```

# Run

```
python app.py
```
