from torch.utils.data import Dataset
import imageio
import logging
import os


class ImageDataset(Dataset):
    def __init__(self, image_path, transforms):
        self.transforms = transforms

        if not (os.path.exists(image_path) and os.path.isdir(image_path)):
            logging.error("invalid path")
            exit(0)

        self.images = []
        for img_path in os.listdir(image_path):
            image = imageio.imread(os.path.join(image_path, img_path))[:, :, :3]
            self.images.append(image)

        self.num_of_images = len(self.images)

        assert self.num_of_images > 0

    def __getitem__(self, idx):
        image = self._get_image(idx)

        return self.transforms(image)

    def _get_image(self, idx):
        return self.images[idx % self.num_of_images]

    def __len__(self):
        return self.num_of_images
