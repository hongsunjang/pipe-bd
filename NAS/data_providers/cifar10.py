# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from data_providers.base_provider import *
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class Cutout(object):
    """	
    randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return  img


class Cifar10DataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08,  distort_color=None, cutout = None, train_method = 'dp', num_replicas = 1):

        self._save_path = save_path

        train_transforms = self.build_train_transform(distort_color, resize_scale, cutout)
        test_transforms = self.build_test_transform()
        train_dataset = CIFAR10(root=self.train_path, train=True, transform=train_transforms, download=True)

        if valid_size is None:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler= train_sampler,
                num_workers=n_worker, pin_memory=True
            )
            self.valid = None
        else:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            
            train_indexes, valid_indexes = self.random_sample_valid_set(
                [cls for _, cls in train_dataset], valid_size, self.n_classes,
            )

            valid_dataset = CIFAR10(root=self.train_path, train=True, transform=test_transforms)

            if train_method == 'ts': 
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            elif torch.distributed.get_rank() >= num_replicas:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            else:
                train_dataset = torch.utils.data.Subset(train_dataset, train_indexes)
                valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indexes)
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = num_replicas,  shuffle=True)
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas = num_replicas, shuffle=False)
            
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler, 
                num_workers=n_worker, pin_memory=True,  # drop_last = True
            )
            if train_method != 'dp':
                n_worker = 1 # efficient architecture validation set load

            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True, # drop_last = True
            )
        
        test_dataset = CIFAR10(root=self.valid_path, train=False, transform=test_transforms)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size = test_batch_size, sampler=test_sampler,
                num_workers=n_worker, pin_memory=True, shuffle =False
        )
        """
        self.train = zip(train_dataset.data, train_dataset.label)
        self.valid = zip(valid_dataset.data, valid_dataset.label)
        self.test = zip(test_dataset.data, valid_dataset.label)
        """
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/hsjang0918/cifar10'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download Cifar10')

    @property
    def train_path(self):
        return self.save_path

    @property
    def valid_path(self):
        return self.save_path

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])

    def build_train_transform(self, distort_color, resize_scale, cutout):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        if cutout is not None:
            print("Data augmentation c.o. is used")
            train_transforms.transforms.append(Cutout(n_holes=1, length=cutout))

        return train_transforms

    def build_test_transform(self):
        test_transforms = transforms.Compose([
            transforms.ToTensor(), 
            self.normalize
        ])
        return test_transforms

    @property
    def resize_value(self):
        return 32

    @property
    def image_size(self):
        return 32
