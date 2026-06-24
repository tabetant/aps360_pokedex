"""Same-species cross-style PAIR batches for contrastive training.

The baseline PokedexDataset yields single images per style -- fine for eval, but
contrastive training needs PAIRS: two styles of the same species. Two rules:
  1. each item = (anchor image, positive image) of the SAME species, different styles.
  2. DISTINCT species within a batch -- the InfoNCE loss treats every other item
     as a negative, so a repeated species would create a false negative.

We pre-build a deterministic "plan" (list of (species, style_a, style_b)) arranged
so each consecutive block of `batch_size` entries is one valid batch, then hand it
to a normal DataLoader (shuffle=False) -- which gives us worker-parallel image
loading for free, mirroring get_data_loader.
"""
import random
import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
from src.pokedex.data.dataset import transform_builder

MODERN = ("render", "gen5", "art")


def _build_species_index(manifest, split="train"):
    """species_id -> {style: image_path}, restricted to one split."""
    idx = {}
    for r in manifest:
        if r["split"] == split:
            idx.setdefault(r["species_id"], {})[r["style"]] = r["image_path"]
    return idx


def _pools(species_index):
    """retro pool = species with retropixel AND >=1 modern style;
    modern pool = species with >=2 modern styles (needed for a modern-modern pair)."""
    retro = [s for s, st in species_index.items()
             if "retropixel" in st and any(m in st for m in MODERN)]
    modern = [s for s, st in species_index.items()
              if sum(m in st for m in MODERN) >= 2]
    return retro, modern


def _make_plan(species_index, batch_size, retro_frac, n_batches, rng):
    retro_sp, modern_sp = _pools(species_index)
    n_retro = round(retro_frac * batch_size)
    n_mod = batch_size - n_retro
    assert len(retro_sp) >= n_retro, (
        f"need {n_retro} distinct retro species/batch but only {len(retro_sp)} exist")

    plan = []
    for _ in range(n_batches):
        retro_pick = rng.sample(retro_sp, n_retro)
        used = set(retro_pick)
        modern_cands = [s for s in modern_sp if s not in used]   # keep batch distinct
        assert len(modern_cands) >= n_mod, "not enough modern species for a distinct batch"
        modern_pick = rng.sample(modern_cands, n_mod)

        for s in retro_pick:                       # retropixel <-> random modern
            pos = rng.choice([m for m in MODERN if m in species_index[s]])
            plan.append((s, "retropixel", pos))
        for s in modern_pick:                      # two distinct modern styles
            a, b = rng.sample([m for m in MODERN if m in species_index[s]], 2)
            plan.append((s, a, b))
    return plan


class PairDataset(Dataset):
    def __init__(self, plan, species_index, transform):
        self.plan = plan
        self.idx = species_index
        self.transform = transform

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, i):
        sid, style_a, style_b = self.plan[i]
        a = self.transform(PIL.Image.open(self.idx[sid][style_a]).convert("RGB"))
        b = self.transform(PIL.Image.open(self.idx[sid][style_b]).convert("RGB"))
        return a, b, sid


def make_pair_loader(manifest, batch_size=64, retro_frac=0.7,
                     steps_per_epoch=None, seed=0, num_workers=4):
    """Build a fresh DataLoader of contrastive pair-batches.

    Call once PER EPOCH with seed=epoch so the species selection reshuffles.
    Yields (anchor_imgs [B,3,256,256], positive_imgs [B,3,256,256], species_ids [B]).
    Each batch already has distinct species and the ~retro_frac retro/modern mix.
    """
    species_index = _build_species_index(manifest, split="train")
    rng = random.Random(seed)
    _, modern_sp = _pools(species_index)
    n_mod = batch_size - round(retro_frac * batch_size)
    if steps_per_epoch is None:                    # ~one pass over the modern pool
        steps_per_epoch = max(1, len(modern_sp) // n_mod)

    plan = _make_plan(species_index, batch_size, retro_frac, steps_per_epoch, rng)
    ds = PairDataset(plan, species_index, transform_builder(train=True))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=True)
