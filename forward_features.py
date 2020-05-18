import os
import random
import pickle
import torch
from datamgr import SimpleDataManager, SetDataManager
from io_utils import parse_args, resume_training, enable_gpu_usage
from backbone import wrn28_10
from top_losses import LossesBag
from losses import get_bag
from graph_mixup import evaluate


use_gpu = torch.cuda.is_available()

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def save_features(model, losses_bag, data_loader, features_dir, params):
    penultimate_dict, features_dict = evaluate(data_loader, model, losses_bag, params, save_latent=True)
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

    n_way, n_shot, n_val = params.n_way, params.n_shot, params.n_val
    novel_datamgr = SetDataManager(novel_file, image_size, n_way, n_shot, n_val)
    novel_loader = novel_datamgr.get_data_loader(aug=False)

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
        save_features(model, losses_bag, novel_loader, features_dir, params)
