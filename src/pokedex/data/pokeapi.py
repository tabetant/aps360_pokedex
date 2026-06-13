import os
import json
import requests

NAMES_ENDPOINT = "pokemon-species"
SPRITES_ENDPOINT = "pokemon"
LANGS = ["en", "ja-hrkt", "ko", "fr", "de", "es", "it"] 
RETRO_GEN1 = ["red-blue", "yellow"]
RETRO_GEN2 = ["gold", "silver", "crystal"]
SPRITES_GEN3 = ["ruby-sapphire", "emerald", "firered-leafgreen"]
SPRITES_GEN4 = ["diamond-pearl", "platinum", "heartgold-soulsilver"]
SPRITES_GEN5 = ["black-white"]

def get(endpoint, poke_id):
    os.makedirs(f"data/cache/{endpoint}", exist_ok=True)
    file_path = f"data/cache/{endpoint}/{poke_id}.json"
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        response = requests.get(f"https://pokeapi.co/api/v2/{endpoint}/{poke_id}")
        if response.status_code == 200:
            data = response.json()
            with open(file_path, "w") as f:
                json.dump(data, f)
            return data
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")

def parse_names_and_gen(species, langs=LANGS):
    names = {}
    generation = -1
    for entry in species["names"]:
        if entry["language"]["name"] in langs:
            names[entry["language"]["name"]] = entry["name"]
            generation = species["generation"]["name"]
    return names, generation

def _front_defaults(pokemon, generation, versions):
    """Non-None front_default URLs for the given game versions under one generation, in order."""
    gen = pokemon["sprites"]["versions"][generation]
    return [gen[v]["front_default"] for v in versions if gen[v]["front_default"] is not None]


def parse_sprites(pokemon):
    retropixel = (
        _front_defaults(pokemon, "generation-i", RETRO_GEN1)
        + _front_defaults(pokemon, "generation-ii", RETRO_GEN2)
    )
    gen5 = (
        _front_defaults(pokemon, "generation-v", SPRITES_GEN5)
        + _front_defaults(pokemon, "generation-iv", SPRITES_GEN4)
        + _front_defaults(pokemon, "generation-iii", SPRITES_GEN3)
    )
    render = pokemon["sprites"]["other"]["home"]["front_default"]
    art = pokemon["sprites"]["other"]["official-artwork"]["front_default"]
    return {
        "retropixel": retropixel,
        "gen5": gen5,
        "render": [render] if render is not None else [],
        "art": [art] if art is not None else [],
    }

def parse(poke_id):
    species =  get(NAMES_ENDPOINT, poke_id)
    names, generation = parse_names_and_gen(species)
    pokemon = get(SPRITES_ENDPOINT, poke_id)
    return {
        "id": poke_id,
        "names": names,
        "generation": generation,
        "sprites": parse_sprites(pokemon)
    }
