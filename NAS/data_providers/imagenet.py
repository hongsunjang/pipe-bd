# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *


class ImagenetDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None, train_method = 'dp', num_replicas=1):

        self._save_path = save_path
        train_transforms = self.build_train_transform(distort_color, resize_scale)
        train_dataset = datasets.ImageFolder(self.train_path, train_transforms)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            
            train_indexes, valid_indexes = self.random_sample_valid_set(
                [cls for _, cls in train_dataset.samples], valid_size, self.n_classes,
            )

            valid_dataset = datasets.ImageFolder(self.train_path, transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                self.normalize,
            ]))
            
            if train_method == 'ts': 
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            elif torch.distributed.get_rank() >= num_replicas:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            else:
                train_dataset = torch.utils.data.Subset(train_dataset, train_indexes)
                valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indexes)
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = num_replicas, shuffle=True)
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas = num_replicas, shuffle=False)
            

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )

            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        test_dataset = datasets.ImageFolder(self.valid_path, transforms.Compose([
            transforms.Resize(self.resize_value),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            self.normalize,
        ]))
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, sampler=test_sampler, 
            num_workers=n_worker, pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            #self._save_path = '~/image'
            self._save_path = '/datasets/image'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 224
