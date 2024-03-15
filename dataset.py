import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FOVDataset(Dataset):
    """
    A dataset class for handling Field of View (FOV) images.

    The class is designed to load images from two directories: 'input' and 'ground truth (gt)',
    where 'input' contains the input images and 'gt' contains the corresponding ground truth images.
    The class supports custom transformations for both input and ground truth images.

    Attributes:
        gt_dir (str): Path to the directory containing ground truth images.
        input_dir (str): Path to the directory containing input images.
        transform: A function/transform that takes in a PIL image and returns a transformed version.
                   Default transformations include converting the image to tensor,
                   converting it to grayscale, and resizing to 512x512 pixels.
        images (list): A sorted list of filenames found in the input directory.

    Args:
        data_dir (str): The base directory path containing 'input' and 'gt' subdirectories.
        transform (optional): An optional transform to be applied on both input and ground truth images.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Retrieves the input and ground truth image at the specified index,
                          applies the transformations, and returns them as a tuple.

    Example:
        train_dataset = FOVDataset('fov_extension_data/train', transform=None)
    """

    def __init__(self, data_dir, transform=None):

        self.gt_dir = os.path.join(data_dir, 'gt')
        self.input_dir = os.path.join(data_dir, 'input')
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((512, 512), antialias=True)
            ])

        self.images = sorted(os.listdir(self.input_dir))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.images[idx])
        gt_path = os.path.join(self.gt_dir, self.images[idx])

        input_image = Image.open(input_path)
        gt_image = Image.open(gt_path)

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image
