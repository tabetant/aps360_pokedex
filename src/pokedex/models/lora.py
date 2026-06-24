"""LoRA-adapted SigLIP 2 image tower for cross-style fine-tuning.

Separate from encoder.py: that module loads a FROZEN model at import for the
baseline (no grad). This one builds a TRAINABLE model on demand -- the whole
backbone is frozen and tiny LoRA adapters are injected into the vision tower's
attention q/v projections, so only <1% of params train and the modern-style
geometry the baseline already learned stays anchored.
"""
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import torch.nn.functional as F

CKPT = "google/siglip2-base-patch16-256"   # same checkpoint as encoder.py


def _freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def _vision_qv_targets(model):
    """Full module names of q_proj / v_proj INSIDE the vision tower only.

    Passing the exact full names (not the bare suffixes) keeps LoRA off the
    text tower, which must stay frozen to match the baseline. 12 encoder
    layers x {q_proj, v_proj} -> expect 24 targets for B/16.
    """
    targets = [n for n, m in model.named_modules()
               if "vision_model" in n and n.endswith(("q_proj", "v_proj"))]
    if not targets:
        raise RuntimeError(
            "Found 0 vision q/v projection modules -- SigLIP's internal naming "
            "differs from expected. Inspect model.named_modules() and adjust.")
    return targets


def build_lora_model(r=8, alpha=16, dropout=0.05, device=None):
    """Frozen SigLIP 2 + LoRA adapters on vision q/v projections. Returns the
    trainable PeftModel already moved to device."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _freeze(AutoModel.from_pretrained(CKPT))
    targets = _vision_qv_targets(model)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=targets,
                     lora_dropout=dropout, bias="none")
    model = get_peft_model(model, cfg)
    model.to(device)
    print(f"LoRA on {len(targets)} modules (r={r}, alpha={alpha}); ", end="")
    model.print_trainable_parameters()          # sanity: should be <1%
    return model


def _image_features(model, image_batch):
    # mirrors encoder.py's working call; PeftModel proxies the attribute down to
    # the wrapped SigLIP. If a transformers version ever breaks the proxy, use
    # model.base_model.model.get_image_features(...) instead.
    return model.get_image_features(image_batch).pooler_output


def encode(model, image_batch, device=None):
    """Training-time encode: GRAD ON. Caller must have set model.train() (so
    LoRA dropout is active). Returns L2-normalized features on `device`."""
    device = device or next(model.parameters()).device
    feats = _image_features(model, image_batch.to(device))
    return F.normalize(feats, p=2, dim=-1)


def make_lora_embed(model, device=None):
    """Return an eval encoder matching embed_image's signature:
    lora_embed(image_batch) -> L2-normalized feats, no grad.

    SIDE EFFECT: flips the model to eval(). After validation, call model.train()
    again before resuming the next training epoch. retrieval.embed_style does the
    .detach().cpu(), so returning device tensors here is fine.
    """
    device = device or next(model.parameters()).device

    def lora_embed(image_batch):
        model.eval()
        with torch.no_grad():
            feats = _image_features(model, image_batch.to(device))
        return F.normalize(feats, p=2, dim=-1)

    return lora_embed


def save_adapters(model, path):
    """Save ONLY the LoRA adapter weights (a few MB) -- checkpoint to Drive."""
    model.save_pretrained(path)


def load_lora_model(path, device=None):
    """Reload a saved adapter on top of a fresh frozen backbone (for resume/eval)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = _freeze(AutoModel.from_pretrained(CKPT))
    model = PeftModel.from_pretrained(base, path)
    model.to(device)
    return model
