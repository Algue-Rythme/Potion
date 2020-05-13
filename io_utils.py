import os
import glob
import argparse
import numpy as np
import torch


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset', default='miniImagenet', help='CUB/miniImagenet')
    parser.add_argument('--run_name', default='default', help='Name of the xp')
    parser.add_argument('--model', default='WideResNet28_10', help='model:  WideResNet28_10')
    parser.add_argument('--train_aug', action='store_true', help='perform data augmentation or not during training ')
    parser.add_argument('--lazy_load', action='store_true', help='Cache the images in RAM')
    parser.add_argument('--num_classes', default=100, type=int, help='total number of classes')
    parser.add_argument('--save_freq', default=10, type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help ='Starting epoch')
    parser.add_argument('--stop_epoch', default=400, type=int, help ='Stopping epoch')
    parser.add_argument('--resume', action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('--lr', default=0.001, type=int, help='learning rate')
    parser.add_argument('--test_batch_size', default=32, type=int, help='batch size ')
    parser.add_argument('--unit_sphere', action='store_true', help='project onto unit sphere')

    if script == 'train':
        parser.add_argument('--batch_size', default=16, type=int, help='batch size ')
        parser.add_argument('--alpha', default=2.0, type=int, help='for S2M2 training ')
    if script == 'graph':
        parser.add_argument('--n_way', default=3, type=int, help='ways')
        parser.add_argument('--n_shot', default=1, type=int, help ='shots')
        parser.add_argument('--n_val', default=1, type=int, help ='vals')
    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    filelist =  [x  for x in filelist if os.path.basename(x) != 'best.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best.tar')
    if os.path.isfile(best_file):
        return best_file
    return get_resume_file(checkpoint_dir)

def resume_training(checkpoint_dir, model):
    resume_file = get_resume_file(checkpoint_dir)        
    print("resume_file", resume_file)
    tmp = torch.load(resume_file)
    start_epoch = tmp['epoch']+1
    print("restored epoch is" , tmp['epoch'])
    state = tmp['state']
    model.load_state_dict(state)
    return start_epoch

def enable_gpu_usage(model):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    return model
    