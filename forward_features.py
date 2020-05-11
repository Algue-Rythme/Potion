import collections
import os
import random
import pickle
import torch
import torch.backends.cudnn as cudnn
import tqdm
from datamgr import SimpleDataManager
from io_utils import parse_args, resume_training, enable_gpu_usage
from backbone import wrn28_10


use_gpu = torch.cuda.is_available()

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def save_features(model, data_loader, features_dir):
    model.eval()
    output_dict = collections.defaultdict(list)
    progress = tqdm.tqdm(total=len(data_loader), leave=True, ascii=True)
    for inputs, targets in data_loader:
        if use_gpu:
            inputs = inputs.cuda()
        out_latent = model(inputs).cpu().numpy()
        for sample, target in zip(out_latent, targets):
            output_dict[int(target.item())].append(sample)
        progress.update()
    progress.close()
    save_pickle(os.path.join(features_dir, '/novel.plk'), output_dict)

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

    if use_gpu:
        model = enable_gpu_usage(model)
        cudnn.benchmark = True
    start_epoch = resume_training(params.checkpoint_dir, model)

    features_dir = 'images/%s/%s/%s/%s/' %(params.dataset, params.model, params.run_name, str(start_epoch))
    if not os.path.isdir(features_dir):
        os.makedirs(features_dir)

    with torch.no_grad():
        save_features(model, novel_loader, features_dir)
