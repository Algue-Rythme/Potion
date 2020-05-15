import collections
import os
import random
import pickle
import torch
import tqdm
from datamgr import SimpleDataManager
from io_utils import parse_args, resume_training, enable_gpu_usage
from backbone import wrn28_10
from top_losses import LossesBag
from losses import get_bag


use_gpu = torch.cuda.is_available()

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def save_features(model, losses_bag, data_loader, features_dir):
    if not params.local_batch:
        model.eval()
        losses_bag.eval()
    penultimate_dict = collections.defaultdict(list)
    features_dict = collections.defaultdict(list)
    progress = tqdm.tqdm(total=len(data_loader), leave=True, ascii=True)
    for inputs, targets in data_loader:
        if use_gpu:
            inputs = inputs.cuda()
        inputs = torch.flatten(inputs, 0, 1)
        targets = torch.flatten(targets, 0, 1)
        penultimate_latent = model(inputs)
        features_latent, desc = losses_bag.agregate_features(penultimate_latent)
        penultimate_latent = penultimate_latent.cpu().numpy()
        for penultimate, features, target in zip(penultimate_latent, features_latent, targets):
            penultimate_dict[int(target.item())].append(penultimate)
            features_dict[int(target.item())].append(features)
        progress.set_description(desc=desc)
        progress.update()
    progress.close()
    save_pickle(os.path.join(features_dir, 'novel_penultimate.plk'), penultimate_dict)
    save_pickle(os.path.join(features_dir, 'novel_features.plk'), features_dict)

if __name__ == '__main__':
    params = parse_args('graph')
    random.seed(914637)
    image_size = 84
    save_dir = './weights'
    data_dir = {'cifar':'./filelists/cifar/',
                'CUB':'./filelists/CUB/',
                'miniImagenet':'./filelists/miniImagenet/'}

    novel_file = data_dir[params.dataset] + 'novel.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s/%s' %(save_dir, params.dataset, params.model, params.run_name)

    novel_datamgr = SimpleDataManager(novel_file, image_size, split_ratio=0., lazy_load=params.lazy_load)
    novel_loader = novel_datamgr.get_data_loader(mode='test', batch_size=params.test_batch_size, aug=False, num_workers=12)

    if params.model == 'WideResNet28_10':
        model = wrn28_10(num_classes=params.num_classes)
    else:
        raise ValueError
    bag = get_bag(params)
    losses_bag = LossesBag(bag)

    if use_gpu:
        model = enable_gpu_usage(model)
        losses_bag.use_gpu()
    start_epoch = resume_training(params.checkpoint_dir, model)
    losses_bag.load_states(params.checkpoint_dir)

    features_dir = 'images/%s/%s/%s/%s/' %(params.dataset, params.model, params.run_name, str(start_epoch))
    if not os.path.isdir(features_dir):
        os.makedirs(features_dir)

    with torch.no_grad():
        save_features(model, losses_bag, novel_loader, features_dir)
