#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import yaml
import torch
import random
import argparse
import numpy as np



# Experiment arguments
parser = argparse.ArgumentParser(description='VQ-APC learning framework')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for reproducable results.', required=False)
parser.add_argument('--task', choices=['self-learning', 'phn-clf', 'spk-clf'],
                    help='Choice of task to be performed', required=True)
parser.add_argument('--mode', choices=['train', 'test'], default='train',
                    help='Test mode will load model and test only', required=False)

# Hardware related
parser.add_argument('--njobs', default=6, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--dev_njobs', default=1, type=int,
                    help='Number of threads for dev set dataloader (used in training mode only)',
                    required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')

# Misc.
parser.add_argument('--logdir', default='log/', type=str,
                    help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str,
                    help='Checkpoint path.', required=False)
parser.add_argument('--draw', action='store_true',
                    help='Plot spectrogram / histogram to tensorboard', required=False)
parser.add_argument('--write-test', action='store_true',
                    help='Store phn classification result.', required=False)
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')

paras = parser.parse_args()
setattr(paras, 'gpu', not paras.cpu)
setattr(paras, 'pin_memory', not paras.no_pin)
setattr(paras, 'verbose', not paras.no_msg)
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)


# For reproducibility, comment these to speed up training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)

# Create Solver to deploy task
if paras.task == 'self-learning':
    # Train speech representation models
    from bin.self_learning import Solver
    assert paras.mode == 'train', 'self-learning does not have testing mode'
elif paras.task in ['phn-clf','spk-clf']:
    from bin.downstream import Solver
else:
    raise NotImplementedError

# Execution
solver = Solver(config, paras)
solver.load_data()
solver.set_model()
solver.exec()
