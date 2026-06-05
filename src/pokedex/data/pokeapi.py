import os
import json
import requests

NAMES_ENDPOINT = "pokemon-species"
SPRITES_ENDPOINT = "pokemon"
LANGS = ["en", "ja-hrkt", "ko", "fr", "de", "es", "it"] 

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

def parse_names(species, langs=LANGS):
    names = {}
    for entry in species["names"]:
        if entry["language"]["name"] in langs:
            names[entry["language"]["name"]] = entry["name"]
    return names

def parse_sprites(pokemon):
    gen1 = pokemon["sprites"]["versions"]["generation-i"]["red-blue"]["front_default"]
    gen5 = pokemon["sprites"]["versions"]["generation-v"]["black-white"]["front_default"] 
    render = pokemon["sprites"]["other"]["home"]["front_default"]
    art = pokemon["sprites"]["other"]["official-artwork"]["front_default"]
    if gen1 is None:
        gen1 = pokemon["sprites"]["other"]["showdown"]["front_default"]
    if gen5 is None:
        gen5 = pokemon["sprites"]["versions"]["generation-ix"]["scarlet-violet"]["front_default"]
    return {
        "gen1": gen1,
        "gen5": gen5,
        "render": render,
        "art": art
    }

def parse(poke_id):
    species = get(NAMES_ENDPOINT, poke_id)
    pokemon = get(SPRITES_ENDPOINT, poke_id)
    return {
        "id": poke_id,
        "names": parse_names(species),
        "sprites": parse_sprites(pokemon)
    }
