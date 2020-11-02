
import torch
import numpy as np

MIN_LR = 1e-4

class Optimizer():
    def __init__(self,parameters,optimizer,lr,decay=1.0,**kwargs):
        # Setup torch optimizer
        self.n_steps = 0
        self.opt_type = optimizer
        self.init_lr = lr
        opt = getattr(torch.optim, optimizer)
        if optimizer == 'SGD':
            self.opt = opt(parameters, lr=lr, momentum=0.9, weight_decay=0.0001)
        else:
            self.opt = opt(parameters, lr=lr)
        self.decay_rate = decay

    def get_opt_state_dict(self):
        return self.opt.state_dict()

    def load_opt_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def pre_step(self, step):
        self.opt.zero_grad()
    
    def decay(self):
        if self.decay_rate<1.0:
            for param_group in self.opt.param_groups:
                param_group['lr'] = max(param_group['lr']*self.decay_rate,MIN_LR)

    def step(self):
        self.opt.step()
        self.n_steps += 1

    def create_msg(self):
        return ['Optim.spec.| Algo. = {}\t| Lr = {}\t'
                .format(self.opt_type, self.init_lr)]
