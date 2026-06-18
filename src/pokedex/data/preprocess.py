import os
from PIL import Image
import requests

def download_image(poke_id, style, url):
    ext = url.split(".")[-1]
    dir_path = f"data/cache/images/{poke_id}"
    file_path = f"{dir_path}/{style}.{ext}"
    if os.path.exists(file_path):
        return file_path
    os.makedirs(dir_path, exist_ok=True)
    tmp_path = f"{file_path}.tmp"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"Failed to download image for {poke_id} ({style}): HTTP {response.status_code}")
            return None
        with open(tmp_path, "wb") as f:
            f.write(response.content)
        # validate it actually decodes as an image BEFORE committing to cache
        with Image.open(tmp_path) as im:
            im.load()
        os.replace(tmp_path, file_path)   # atomic: real path only appears once the file is complete + valid
        return file_path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"Failed to download image for {poke_id} ({style}): {e}")
        return None
    
def load_image(path):
    if os.path.exists(path):
        return Image.open(path).convert("RGBA")
    else:
        print(f"Image not found: {path}")
        return None
    
def flatten_white(img):
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")

def to_square_256(img):
    side = max(img.width, img.height)
    new_img = Image.new("RGB", (side, side), (255, 255, 255))
    new_img.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return new_img.resize((256, 256), Image.Resampling.BICUBIC)

def pre_process(poke_id, style, url):
    if url is None:
        return None
    for u in url:
        path = download_image(poke_id, style, u)
        if path is None:
            continue
        img = load_image(path)
        if img is None:
            continue
        img = flatten_white(img)
        img = to_square_256(img)
        return img
    return None