from abc import abstractmethod
from io_utils import get_resume_file
import numpy as np
import torch

def metric_desc(name, losses, correct_total=None):
    avg_loss = np.mean(losses)
    desc = name+('_loss=%.3f'%avg_loss)
    if correct_total is not None:
        correct, total = correct_total
        acc = 100.*correct/total
        desc += ' '+name+('_acc=%.2f%%'%acc)
    return desc

def update_acc(correct_total, outputs, targets):
    correct, total = correct_total
    if len(outputs.shape) == 1:  # binary logit
        predictions = outputs > 0.
    else:
        _, predictions = torch.max(outputs, 1)
    correct += int(predictions.eq(targets).cpu().sum().item())
    total += int(targets.size(0))
    return correct, total

class LossEngine(torch.nn.Module):
    def __init__(self, name, accuracy):
        self.name = name
        super(LossEngine, self).__init__()
        self.losses_items = []
        self.acc_meta = None
        self.last_latent = None
        if accuracy:
            self.acc_meta = 0, 0

    def update_acc(self, outputs, targets):
        self.acc_meta = update_acc(self.acc_meta, outputs, targets)

    def get_desc(self):
        return metric_desc(self.name, self.losses_items, self.acc_meta)

    def clear_epoch(self):
        self.losses_items = []
        if self.acc_meta is not None:
            self.acc_meta = 0, 0

    @property
    def latent(self):
        return self.last_latent

    def record_latent(self, x_latent):
        self.last_latent = x_latent.detach().cpu().numpy()

    @abstractmethod
    def get_features(self):
        pass

    @abstractmethod
    def forward(self):
        pass

class LossesBag:

    def __init__(self, args):
        self.losses_engines = dict()
        for arg in args:
            self.add_loss_engine(arg)

    def __getitem__(self, name):
        return self.losses_engines[name]

    def get_loss_engine(self, name):
        return self.losses_engines[name]

    def add_loss_engine(self, loss_engine):
        self.losses_engines[loss_engine.name] = loss_engine

    def train(self):
        for loss_engine in self.losses_engines.values():
            loss_engine.train()

    def eval(self):
        for loss_engine in self.losses_engines.values():
            loss_engine.eval()

    def optimizer_dict(self):
        return [{'params': loss.parameters()} for loss in self.losses_engines.values()]

    def use_gpu(self):
        for loss_engine in self.losses_engines.values():
            loss_engine.cuda()

    def clear_epoch(self):
        for loss_engine in self.losses_engines.values():
            loss_engine.clear_epoch()

    def get_losses(self, x_latent, target):
        for loss_engine in self.losses_engines.values():
            loss, x_latent = loss_engine(x_latent, target)
            desc = loss_engine.get_desc()
            yield loss_engine.name, loss, x_latent, desc

    def load_states(self, checkpoint_dir):
        resume_file = get_resume_file(checkpoint_dir)
        tmp = torch.load(resume_file)
        for key in tmp:
            if key in ['epoch', 'state']:
                continue
            state = tmp[key]
            self.losses_engines[key].load_state_dict(state)

    def states_dict(self):
        return {loss.name:loss.state_dict() for loss in self.losses_engines.values()}

    def agregate_features(self):
        aggregated = []
        for loss_engine in self.losses_engines.values():
            aggregated.append(loss_engine.latent)
        aggregated = np.concatenate(aggregated, axis=1)  # bigger features
        return aggregated
