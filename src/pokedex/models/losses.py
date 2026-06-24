"""Contrastive loss for cross-style alignment."""
import torch
import torch.nn.functional as F


def info_nce(anchor, positive, temperature=0.07):
    """Symmetric InfoNCE (CLIP-style) over a batch of aligned pairs.

    anchor, positive : (B, D) L2-normalized features; row i of each is the SAME
                       species (different styles). Requires DISTINCT species per
                       batch so off-diagonal entries are true negatives.

    It's cross-entropy over a B x B similarity matrix: for
    anchor i, the "correct class" is column i (its positive); every other column
    is a negative to push away. Done both directions (anchor->positive and
    positive->anchor) and averaged. Lower temperature = sharper contrast.
    """
    logits = (anchor @ positive.T) / temperature          # (B, B) cosine sims, scaled
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) +
                  F.cross_entropy(logits.T, targets))
