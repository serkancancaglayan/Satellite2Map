import os
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MapDataset(Dataset):
    def __init__(self, img_dir, img_size, transform):
        super(MapDataset, self).__init__()
        self.img_list = [os.path.join(img_dir, img_name) for img_name in sorted(os.listdir(img_dir))]   
        self.img_size = img_size
        self.transform = transform


    def __len__(self):
        return len(self.img_list)
    
    """
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = cv.imread(img_path)
        _, img_width = img.shape[:2]
        width_cutoff = img_width // 2
        input_img = cv.resize(img[:, :width_cutoff], dsize=(self.img_size, self.img_size))
        target_img = cv.resize(img[:, width_cutoff:], dsize=(self.img_size, self.img_size))
        return input_img, target_img
    """

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = np.array(Image.open(img_path))
        _, img_width = img.shape[:2]
        width_cutoff = img_width // 2

        input_image = Image.fromarray(img[:, :width_cutoff, :])
        target_image = Image.fromarray(img[:, width_cutoff:, :])
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image





