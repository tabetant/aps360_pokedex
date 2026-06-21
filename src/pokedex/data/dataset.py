import PIL.Image
import torch
import torchvision.transforms as transforms
class PokedexDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split, transform, style):
        self.styledict = {"render":0, "gen5":1, "retropixel":2, "art":3}
        self.manifest = [x for x in manifest if x['style'] == style] if split =='all' else [x for x in manifest if x['split'] == split and x['style'] == style] 
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        
        item = self.manifest[idx]
        image = PIL.Image.open(item['image_path']).convert('RGB')
        species_id = item['species_id']
        style_id = self.styledict[item['style']]

        if self.transform:
            image = self.transform(image)

        return image, species_id, style_id

def transform_builder(train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
def get_data_loader(manifest, split, style, batch_size, num_workers=4):
    transform = transform_builder(train=(split=="train"))
    dataset = PokedexDataset(manifest, split, transform, style)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)