from abc import abstractmethod
import collections
import json
import random
import torch
import torchvision.transforms as transforms
import additional_transforms as add_transforms
from dataset import SimpleDataset, SetDataset, EpisodicBatchSampler


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param = None,
                 jitter_param = None):
        if normalize_param is None:
            normalize_param = dict(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
        if jitter_param is None:
            jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size)
        if transform_type=='CenterCrop':
            return method(self.image_size)
        if transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        if transform_type=='Normalize':
            return method(**self.normalize_param)
        return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']  WHY RESIZE ???
            # transform_list = ['CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

def split_train_test(data_file, split_ratio):
    with open(data_file, 'r') as f:
        meta = json.load(f)
    classes = collections.defaultdict(list)
    for index, label in enumerate(meta['image_labels']):
        classes[label].append(index)
    train_indexes, test_indexes = [], []
    for label in classes:
        train_size = int(split_ratio * len(classes[label]))
        random.shuffle(classes[label])
        train_indexes += classes[label][:train_size]
        test_indexes += classes[label][train_size:]
    return train_indexes, test_indexes

class SimpleDataManager(DataManager):
    def __init__(self, data_file, image_size, split_ratio=1., lazy_load=False):
        super(SimpleDataManager, self).__init__()
        self.data_file = data_file
        self.lazy_load = lazy_load
        self.trans_loader = TransformLoader(image_size)
        train_indexes, test_indexes = self.split_train_test(split_ratio)
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes

    def get_data_loader(self, mode, batch_size, aug, num_workers=8, lazy_load=False):
        transform = self.trans_loader.get_composed_transform(aug)
        indexes = self.train_indexes if mode == 'train' else self.test_indexes
        shuffle = mode == 'train'  # no shuffle at test time
        dataset = SimpleDataset(self.data_file, transform, indexes=indexes, lazy_load=lazy_load)
        data_loader_params = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class SetDataManager(DataManager):
    def __init__(self, data_file, image_size, n_way, n_support, n_query, n_episode):
        super(SetDataManager, self).__init__()
        self.data_file = data_file
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug, num_workers=12):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
