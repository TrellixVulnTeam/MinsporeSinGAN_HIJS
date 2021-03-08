import os
import torchvision.transforms as transforms
from datasets.photoimage import PhotoData


def get_dataset(dataset, args):
        print('USE PHOTO DATASET')
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
        transforms_train = transforms.Compose([transforms.Resize((args.img_size_max, args.img_size_max)),
                                                transforms.ToTensor(),
                                                normalize])
        transforms_val = transforms.Compose([transforms.Resize((args.img_size_max, args.img_size_max)),
                                                transforms.ToTensor(),
                                                normalize])

        train_dataset = PhotoData(args.data_dir, True, transform=transforms_train, img_to_use=args.img_to_use)
        val_dataset = PhotoData(args.data_dir, False, transform=transforms_val, img_to_use=args.img_to_use)

        if train_dataset.randidx != -999:
                args.img_to_use = train_dataset.randidx

        return train_dataset, val_dataset
