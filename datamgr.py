from abc import abstractmethod
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
            method = add_transforms.ImageJitter( self.jitter_param )
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
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug, num_workers=8, lazy_load=False):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform, lazy_load=lazy_load)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_episode=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug, num_workers=12, lazy_load=False):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(data_file, self.batch_size, transform, lazy_load=lazy_load)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=num_workers, pin_memory=True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
