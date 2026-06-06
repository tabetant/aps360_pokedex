import os
import json
from src.pokedex.data.pokeapi import parse
from src.pokedex.data.preprocess import pre_process
import random
from collections import defaultdict

def split_species(records, seed=42):
    rng = random.Random(seed)
    seen = set()
    species_list = defaultdict(list)

    for species in records:
        if species["species_id"] not in seen:
            gen = species["generation"]
            species_list[gen].append(species["species_id"])
            seen.add(species["species_id"])
    train, val, test = [], [], []
    n_train, n_val, n_test = 0, 0, 0
    for gen in species_list:
        rng.shuffle(species_list[gen])
        n = len(species_list[gen])
        n_train_gen = int(0.7 * n)
        n_val_gen = int(0.15 * n)
        train.extend(species_list[gen][:n_train_gen])
        val.extend(species_list[gen][n_train_gen:n_train_gen + n_val_gen])
        test.extend(species_list[gen][n_train_gen + n_val_gen:])
        n_train += n_train_gen
        n_val += n_val_gen
        n_test += (n - n_train_gen - n_val_gen)
    print(f"Train: {n_train} species, Val: {n_val} species, Test: {n_test} species")
    split_map = dict()
    split_map["train"] = set(train)
    split_map["val"] = set(val)
    split_map["test"] = set(test)
    return split_map

def build_dataset(start=1, end=5):
    records = []
    for species_id in range(start, end + 1):
        download_path=f"data/processed/{species_id}"
        os.makedirs(download_path, exist_ok=True)
        try:
            data = parse(species_id)
            for style, url in data["sprites"].items():
                img = pre_process(species_id, style, url)
                img_path = f"{download_path}/{style}.png"
                if img:
                    img.save(img_path)
                    records.append({
                        "species_id": species_id,
                        "generation": data["generation"],
                        "style": style,
                        "image_path": img_path,
                        "names": data["names"],
                })
            print(f"Processed species {species_id}")
        except Exception as e:
            print(f"Error processing species {species_id}: {e}")
    split = split_species(records)
    for record in records:
        sid = record["species_id"]
        for split_name in ("train","val","test"):
            if sid in split[split_name]: 
                record["split"] = split_name
                break     
    return records

def save_dataset(records, output_path="data/processed/manifest.json"):
    with open(output_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset with {len(records)} records to {output_path}")