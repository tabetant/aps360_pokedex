"""Throwaway probe: discover the transformers 5.x SigLIP 2 API + its normalization stats.
Prints facts only. Delete after M5 gate is resolved."""

import torch
import transformers
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np

CKPT = "google/siglip2-base-patch16-256"  # "B/16 256px" — confirm this resolves

print(f"transformers version: {transformers.__version__}")
print(f"checkpoint: {CKPT}\n")

# --- Gate 1a: do AutoModel / AutoProcessor resolve it, and to WHAT classes? ---
processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModel.from_pretrained(CKPT)
model.eval()
print(f"model class:     {type(model).__name__}")
print(f"processor class: {type(processor).__name__}")
print(f"image_processor: {type(processor.image_processor).__name__}\n")

# --- Gate 2: the normalization stats you must feed SigLIP (the payoff) ---
ip = processor.image_processor
print("NORMALIZATION STATS (compare against your .485/.456/.406 placeholder):")
print(f"  image_mean: {getattr(ip, 'image_mean', 'N/A')}")
print(f"  image_std:  {getattr(ip, 'image_std', 'N/A')}")
print(f"  size:       {getattr(ip, 'size', 'N/A')}")
print(f"  resample:   {getattr(ip, 'resample', 'N/A')}")
print(f"  rescale:    {getattr(ip, 'rescale_factor', 'N/A')}\n")

# --- Gate 1b: how do you get an image embedding, and what shape is it? ---
dummy = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
inputs = processor(images=dummy, return_tensors="pt")
print(f"processor output keys: {list(inputs.keys())}")
print(f"pixel_values shape:    {inputs['pixel_values'].shape}\n")

with torch.no_grad():
    feats = model.get_image_features(**inputs)   # if this errors, the API differs — read the traceback
print(f"returned type:  {type(feats).__name__}")
print(f"available keys: {list(feats.keys())}")
for k in feats.keys():
    v = feats[k]
    if hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)}  L2norm={v.float().norm(dim=-1).mean().item():.4f}")