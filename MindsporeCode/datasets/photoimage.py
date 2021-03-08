import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from glob import glob

def pil_loader(path, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            return img.convert('L')
        else:
            return img.convert('RGB')

class PhotoData(data.Dataset):
    def __init__(self, data_dir, is_train, transform=None, singlemode=True, img_to_use=-999):
        self.data_dir = os.path.join(data_dir, 'SinGANdata')
        self.transform = transform
        self.singlemode = singlemode
        self.randidx = -999

        if is_train:
            self.image_dir = os.path.join(self.data_dir, 'trainPhoto')
        else:
            self.image_dir = os.path.join(self.data_dir, 'testPhoto')

        # print('img dir ------ {}'.format(self.image_dir))
        # print('joined then ------- {}'.format(os.path.join(self.image_dir, '*.jpg')))
        # print('glob result ----- {}'.format(glob(os.path.join(self.image_dir, '*.jpg'))))

        if singlemode:
            if img_to_use == -999:
                self.num_data = len(glob(os.path.join(self.image_dir, '*.jpg')))
                randidx = np.random.randint(0, self.num_data)
                self.randidx = randidx
                self.image_dir = sorted(glob(os.path.join(self.image_dir, '*.jpg')))[randidx]
            else:
                self.image_dir = sorted(glob(os.path.join(self.image_dir, '*.jpg')))[img_to_use]
        else:
            self.image_dir = sorted(glob(os.path.join(self.image_dir, '*.jpg')))

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        if self.singlemode:
            img = pil_loader(self.image_dir)
        else:
            img = pil_loader(self.image_dir[idx])

        if self.transform:
            img = self.transform(img)

        return img
