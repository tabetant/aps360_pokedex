# aps360_pokedex

**Multimodal Pokédex: cross-style alignment with SigLIP 2** — APS360 (University of Toronto), Summer 2026.

Aligns a Pokémon species' Gen 1 sprite, Gen 5 sprite, Gen 9 3D model, official artwork, and
multilingual names in one shared embedding space, by LoRA fine-tuning a frozen SigLIP 2 backbone.
This enables **cross-style retrieval**: find every depiction of a species from any one of them.

## Results

| Setting | Top 1 accuracy |
| :--- | :--- |
| Modern artwork, zero shot | 0.94 |
| Retro sprites, zero shot | 0.48 |
| Retro sprites, after LoRA on 0.08% of a 375M model | 0.68 |

Fine tuning the hard domain cost nothing on the strong domains: zero forgetting across styles. Dataset: 23K image pairs, 1,025 species, names in 7 languages.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data is fetched at runtime from [PokéAPI](https://pokeapi.co) — no assets are bundled with this repo.

## Structure
- `src/pokedex/` — importable package (data pipeline, models, eval)
- `notebooks/` — exploration & sanity checks
- `scripts/` — runnable entry points

## Disclaimer

Unaffiliated fan research project. Pokémon and all related names and images are property of Nintendo, Creatures Inc., and Game Freak.
