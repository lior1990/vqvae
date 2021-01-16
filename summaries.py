import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))

    def add_scalar(self, log_name, value, index):
        self.writer.add_scalar(log_name, value, index)

    def visualize_image(self, global_step, images, name, n_images=3):
        grid_image = make_grid(images[:n_images, :, :, :].clone().cpu().data, n_images, normalize=True)
        img_name = f'Image/{name}'
        self.writer.add_image(img_name, grid_image, global_step)
