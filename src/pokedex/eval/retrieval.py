from src.pokedex.models import embed_image
import torch
import os
from src.pokedex.data.dataset import get_data_loader

CACHE_DIR = 'data/cache/embeddings'


def embed_style(loader, style, split, embed_fn=embed_image, use_cache=True):
    """Embed every image of one style/split into (feats, labels).

    embed_fn : callable(image_batch) -> (B, D) L2-normalized features.
               Defaults to the FROZEN baseline encoder (embed_image).
    use_cache: read/write data/cache/embeddings/{style}_{split}.pt.

    WARNING: the on-disk cache is keyed ONLY by style+split, and it currently
    holds the FROZEN M5 baseline embeddings. Pass use_cache=False whenever
    embed_fn is a fine-tuned (LoRA) model, otherwise you will read stale
    baseline features (and silently "see no change") or overwrite the baseline.
    """
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        feat_path = f'{CACHE_DIR}/{style}_{split}.pt'
        label_path = f'{CACHE_DIR}/{style}_{split}_labels.pt'
        if os.path.exists(feat_path):
            return torch.load(feat_path), torch.load(label_path)

    feats = []
    labels = []
    for images, species_ids, style_ids in loader:
        # .detach().cpu() so a GPU LoRA model and the CPU baseline both land
        # on CPU -> retrieve/score_cell stay device-consistent with labels.
        feats.append(embed_fn(images).detach().cpu())
        labels.append(species_ids)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    if use_cache:
        torch.save(feats, feat_path)
        torch.save(labels, label_path)
    return feats, labels


def retrieve(query_feats, gallery_feats, top_k=5):
    sims = query_feats @ gallery_feats.T
    top_k_indices = torch.topk(sims, k=top_k, dim=-1).indices
    return top_k_indices


def score_cell(query_feats, query_labels, gallery_feats, gallery_labels):
    mask = torch.isin(query_labels, gallery_labels)   # (nq,) bool
    query_feats  = query_feats[mask]
    query_labels = query_labels[mask]
    top1_indices = retrieve(query_feats, gallery_feats, 1)
    recall5_indices = retrieve(query_feats, gallery_feats, 5)

    top1_labels = gallery_labels[top1_indices]
    recall5_labels = gallery_labels[recall5_indices]

    top1 = (top1_labels == query_labels.unsqueeze(1)).any(dim=1).float().mean()
    recall5 = (recall5_labels == query_labels.unsqueeze(1)).any(dim=1).float().mean()

    return top1, recall5, len(query_labels)


def build_grid(manifest, embed_fn=embed_image, query_split='test',
               use_cache=True, batch_size=64):
    """Cross-style retrieval grid.

    query_split: 'test' for the final report, 'val' for per-epoch model
                 selection during M6 (NEVER touch test until the end).
    embed_fn / use_cache: pass the LoRA encoder + use_cache=False to evaluate a
                 fine-tuned model on freshly computed embeddings.

    The gallery ('all' split) for each style is encoded once and reused across
    the query loop -- without this, live (use_cache=False) eval would re-encode
    all 1025 gallery images 3x per style, every epoch.
    """
    styles = ['art', 'gen5', 'retropixel', 'render']
    gallery_cache = {}
    grid = []
    for query_style in styles:
        q_feats, q_labels = embed_style(
            get_data_loader(manifest=manifest, style=query_style,
                            split=query_split, batch_size=batch_size),
            query_style, query_split, embed_fn, use_cache)
        for gallery_style in styles:
            if query_style == gallery_style:
                continue
            if gallery_style not in gallery_cache:
                gallery_cache[gallery_style] = embed_style(
                    get_data_loader(manifest=manifest, style=gallery_style,
                                    split='all', batch_size=batch_size),
                    gallery_style, 'all', embed_fn, use_cache)
            g_feats, g_labels = gallery_cache[gallery_style]
            top1, recall5, count = score_cell(q_feats, q_labels, g_feats, g_labels)
            grid.append((query_style, gallery_style, top1.item(), recall5.item(), count))
    return grid


def summarize_grid(grid):
    """Collapse a grid into the M6 pass/fail numbers (locked criteria).

    Retro cells = any cell touching 'retropixel'; modern cells = the rest.
    Returns the means + worst cells you gate on:
      retro_mean >= 0.65  AND  modern_mean >= 0.90  AND  modern_min >= 0.83
    """
    retro  = [t for (qs, gs, t, r, n) in grid if 'retropixel' in (qs, gs)]
    modern = [t for (qs, gs, t, r, n) in grid if 'retropixel' not in (qs, gs)]
    return {
        'retro_mean':  sum(retro) / len(retro),
        'retro_min':   min(retro),
        'modern_mean': sum(modern) / len(modern),
        'modern_min':  min(modern),
        'n_retro_cells':  len(retro),
        'n_modern_cells': len(modern),
    }
