import os
from PIL import Image
import requests

def download_image(poke_id, style, url):
    file_path = f"data/cache/images/{poke_id}/{style}.{url.split('.')[-1]}"
    if os.path.exists(file_path):
        return file_path
    os.makedirs(f"data/cache/images/{poke_id}", exist_ok=True)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
                return file_path
        else:
            raise Exception(f"Failed to download image: {response.status_code}")
    except Exception as e:
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
    path = download_image(poke_id, style, url)
    if path is None:
        return None
    img = load_image(path)
    if img is None:
        return None
    img = flatten_white(img)
    img = to_square_256(img)
    return img