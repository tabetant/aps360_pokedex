from src.pokedex.models import embed_image
import torch
import os 
from src.pokedex.data.dataset import get_data_loader

def embed_style(image_batch, style, split):
    if not os.path.exists('data/cache/embeddings'):
        os.makedirs('data/cache/embeddings')
    feat_path = f'data/cache/embeddings/{style}_{split}.pt'
    label_path = f'data/cache/embeddings/{style}_{split}_labels.pt'
    if os.path.exists(feat_path):
        return torch.load(feat_path), torch.load(label_path)
    feats = []
    labels = []
    for images, species_ids, style_ids in image_batch:
        feat = embed_image(images)
        feats.append(feat)
        labels.append(species_ids)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
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

def build_grid(manifest):
    grid = []
    for query_style in ['art', 'gen5', 'retropixel' , 'render']:
        for gallery_style in ['art', 'gen5', 'retropixel' , 'render']:
            if query_style == gallery_style:
                continue
            query_feats, query_labels = embed_style(get_data_loader(manifest=manifest, style=query_style, split='test', batch_size=64), query_style, split='test')
            gallery_feats, gallery_labels = embed_style(get_data_loader(manifest=manifest, style=gallery_style, split='all', batch_size=64), gallery_style, split='all')
            top1, recall5, count = score_cell(query_feats, query_labels, gallery_feats, gallery_labels)
            grid.append((query_style, gallery_style, top1.item(), recall5.item(), count))
    return grid