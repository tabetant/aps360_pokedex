from transformers import AutoModel, AutoProcessor
import torch
import torch.nn.functional as F

CKPT = "google/siglip2-base-patch16-256"

processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModel.from_pretrained(CKPT)
model.eval()

def embed_image(image_batch):
    with torch.no_grad():
        feats = model.get_image_features(image_batch).pooler_output
    feats = F.normalize(feats, p=2, dim=-1)
    return feats