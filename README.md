# aps360_pokedex

**Multimodal Pokédex: cross-style alignment with SigLIP 2** — APS360 (University of Toronto), Summer 2026.

Aligns a Pokémon species' Gen 1 sprite, Gen 5 sprite, Gen 9 3D model, official artwork, and
multilingual names in one shared embedding space, by LoRA fine-tuning a frozen SigLIP 2 backbone.
This enables **cross-style retrieval**: find every depiction of a species from any one of them.

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
