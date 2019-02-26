# online data loader

import os
import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
from torch.utils import data
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def pil_loader(path, mode):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if mode == "L":
                return img.convert('L')
            if mode == "RGB":
                return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, mode="RGB"):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dirs):

    classes = []
    for dir in dirs:
        classes.extend([int(d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])

    classes.sort()
    class_to_idx = {i: i for i in classes}
    return classes, class_to_idx


def make_dataset(dirs, class_to_idx):

    images = []
    for dir in dirs:
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            y = int(target)
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[y])
                        images.append(item)

    return images


class CostumeImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader, mode="RGB"):
        classes, class_to_idx = find_classes(roots)
        imgs = make_dataset(roots, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders",
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.roots = roots
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    roots = ['./data/mnist_png/training/0/', './data/mnist_png/training/1/']
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = CostumeImageFolder(roots, transform=transform_train)
    trainLoader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=False, num_workers=4)
    for x, target in trainLoader:
        print(x.size())
        print(target)