import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(path):
    image = cv2.imread(path)
    if image is None or len(image.shape) < 2 or image.shape[2] != 3:
        image = np.zeros((100, 100, 3))
    return image


class ZindiDataset(Dataset):
    def __init__(self, img_folder, train_csv, test_fold_number, is_test, transform):
        dataset = pd.read_csv(train_csv)
        if is_test:
            dataset = dataset[dataset.fold == test_fold_number]
        else:
            dataset = dataset[dataset.fold != test_fold_number]
        self.root = img_folder
        self.image_paths = dataset['image_path'].values
        self.images = [load_image(os.path.join(self.root, image_path)) for image_path in tqdm(self.image_paths)]
        self.targets = dataset['target'].values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        target = self.targets[index]
        image = self.images[index]
        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            'images': image,
            'targets': target,
        }


class ZindiInferenceDataset(Dataset):
    def __init__(self, img_folder, transform):
        self.root = img_folder
        self.image_paths = os.listdir(img_folder)
        self.image_ids = [path.replace('.png', '') for path in self.image_paths]
        self.images = [load_image(os.path.join(self.root, image_path)) for image_path in tqdm(self.image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self.images[index]
        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            'image_id': image_id,
            'image': image,
        }

