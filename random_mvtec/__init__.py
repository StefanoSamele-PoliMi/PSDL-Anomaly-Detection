import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, dataset_path, class_name='bottle', is_train=True, resize=256, cropsize=224):
        assert class_name in CLASS_NAMES+['all'], 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset; you can either load a specific class or the whole dataset
        if class_name == 'all':
            self.x, self.y, self.mask = self.load_all_dataset()
        else:
            self.x, self.y, self.mask = self.load_dataset_folder()


        # set transforms
        self.transform_resize = T.Compose([T.Resize(resize, InterpolationMode.BICUBIC), T.ToTensor()])
        self.random_mvtec_rotation = RandomRotation(10)
        self.random_mvtec_crop = RandomCrop(224)
        self.transform_normalize_x = T.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):

        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_resize(x)

        if y == 0:
            mask = Image.new('L', (self.resize, self.resize))
        else:
            mask = Image.open(mask)

        mask = self.transform_resize(mask)

        x, mask = self.random_mvtec_rotation(x, mask)
        x, mask = self.random_mvtec_crop(x, mask)

        x = self.transform_normalize_x(x)

        return x, y, mask.to(torch.int8)

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

    def load_all_dataset(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        for mvtec_class in os.listdir(self.dataset_path):
            img_dir = os.path.join(self.dataset_path, mvtec_class, phase)
            gt_dir = os.path.join(self.dataset_path, mvtec_class, 'ground_truth')

            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:

                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
                x.extend(img_fpath_list)

                # load gt labels
                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                     for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, mask):

        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        return image[:, top: top + new_h, left: left + new_w], mask[:, top: top + new_h, left: left + new_w]


class RandomRotation(object):
    """Randomly rotate the image between a selected range of angles.

    Args:
        degrees (tuple or int): Desired angles range. If int, range is assumed to be -degrees,+degrees
    """

    def __init__(self, degrees):
        assert isinstance(degrees, (int, tuple))
        if isinstance(degrees, int):
            self.degrees = (-degrees, degrees)
        else:
            assert len(degrees) == 2
            self.degrees = degrees

    def __call__(self, image, mask):

        angle = (torch.rand(1)*(self.degrees[1]-self.degrees[0]) + self.degrees[0]).item()

        rotated_image = rotate(image, angle, InterpolationMode.NEAREST)
        rotated_mask = rotate(mask, angle, InterpolationMode.NEAREST)

        return rotated_image, rotated_mask


