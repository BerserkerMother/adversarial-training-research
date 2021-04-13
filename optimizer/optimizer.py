import numpy as np
import math


class Linear_Warmup_Wrapper:
    def __init__(self, optimizer, warmup_steps: int = 400, max_lr: float = 1e-2):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.total_steps = 1000

        self.current_step = 0.

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        if self.current_step < self.warmup_steps:
            lr = self.current_step / self.warmup_steps * self.max_lr
        else:
            lr = (1 - (self.current_step / self.total_steps)) * self.max_lr

        self.current_step += 1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self):
        self.update_lr()
        self.optimizer.step()

    def get_current_lr(self):
        lrS = []
        for param_group in self.optimizer.param_groups:
            lrS.append(param_group['lr'])

        return lrS

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'warm_steps': self.warmup_steps,
            'current_steps': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.optimizer = state_dict['optimizer']
        self.warmup_steps = state_dict['warm_steps']
        self.current_step = state_dict['current_steps']


class ScheduledOptim():
    # code from attention is all you implementation
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps=1000):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        lrS = []
        for param_group in self._optimizer.param_groups:
            lrS.append(param_group['lr'])

        return lrS

    def state_dict(self):
        return {
            'optimizer': self._optimizer.state_dict(),
            'warm_steps': self.n_warmup_steps,
            'current_steps': self.n_current_steps
        }

    def load_state_dict(self, state_dict):
        self._optimizer = state_dict['optimizer']
        self.n_current_steps = state_dict['warm_steps']
        self.n_current_steps = state_dict['current_steps']


class Cosine_Warmup_Wrapper:
    def __init__(self, optimizer, lr: float, total_steps: int = 4000):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.current_step = 0
        self.init_lr = lr

    def step(self):
        scale = self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.init_lr * scale

        self.current_step += 1

    def get_lr_scale(self):
        scale = .5 * (math.cos(self.current_step / self.total_steps * math.pi) + 1)

        return scale

    def get_current_lr(self):
        lrS = []
        for param_group in self.optimizer.param_groups:
            lrS.append(param_group['lr'])

        return lrS

    def state_dict(self):
        return {
            'current_steps': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_steps']
