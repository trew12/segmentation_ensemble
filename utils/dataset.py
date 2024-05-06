from torch.utils.data import Dataset
import os
from PIL import Image


class CarSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_images = len(os.listdir(img_dir))

    def __len__(self):
        return self.num_images

    def __getitem__(self, i):
        filename  = f"{i:06d}.png"
        data = {'image': Image.open(os.path.join(self.img_dir, filename)),
                'mask': Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
                'filename': filename}
        return data


def collate_fn(batch):
    return [data['image'] for data in batch], [data['mask'] for data in batch], [data['filename'] for data in batch]
