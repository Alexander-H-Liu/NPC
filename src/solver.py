import os
import sys
import abc
import math
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from config.default_hparas import default_hparas
from src.util import human_format, Timer

class BaseSolver():
    ''' 
    Prototype Solver for all kinds of tasks
    Arguments
        config - yaml-styled config
        paras  - argparse outcome
        mode - string that specifies training/testing
    '''

    def __init__(self, config, paras):
        # General Settings
        self.config = config
        self.paras = paras
        for k, v in default_hparas.items():
            setattr(self, k, v)
        if self.paras.gpu and torch.cuda.is_available():
            self.gpu = True
            self.device = torch.device('cuda')
        else:
            self.gpu = False
            self.device = torch.device('cpu')

        # Settings for training/testing
        self.mode = self.paras.mode # legacy, should be removed

        # Name experiment
        self.exp_name = paras.name
        if self.exp_name is None:
            # By default, exp is named after config file
            self.exp_name = paras.config.split('/')[-1].split('.y')[0]
            self.exp_name += '_sd{}'.format(paras.seed)
                
        # Filepath setup
        os.makedirs(paras.ckpdir, exist_ok=True)
        self.ckpdir = os.path.join(paras.ckpdir, self.exp_name)
        os.makedirs(self.ckpdir, exist_ok=True)

        # Logger settings
        self.logdir = os.path.join(paras.logdir, self.exp_name)
        self.log = SummaryWriter(
            self.logdir, flush_secs=self.TB_FLUSH_FREQ)
        self.timer = Timer()

        # Hyperparameters
        self.step = 0
        self.epoch = config['hparas']['epoch']
        
        self.verbose('Exp. name : {}'.format(self.exp_name))
        self.verbose('Loading data...')

    def backward(self, loss):
        '''
        Standard backward step with timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.GRAD_CLIP)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def load_ckpt(self):
        ''' Load ckpt if --load option is specified '''
        if self.paras.load:
            # Load weights
            ckpt = torch.load( self.paras.load,
                map_location=self.device if self.paras.mode == 'train' else 'cpu')
            ckpt['model'] = {k.replace('module.','',1):v \
                                for k,v in ckpt['model'].items()}
            self.model.load_state_dict(ckpt['model'])

            # Load task-dependent items
            metric = "None"
            score = 0.0
            for k, v in ckpt.items():
                if type(v) is float:
                    metric, score = k, v
            if self.paras.mode == 'train':
                self.cur_epoch = ckpt['epoch']
                self.step = ckpt['global_step']
                self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                msg = \
                    'Load ckpt from {}, restarting at step {} \
                    (recorded {} = {:.2f} %)'\
                    .format(self.paras.load, self.step, metric, score)
                self.verbose(msg)
            else:
                # Inference
                msg = 'Evaluation target = {} (recorded {} = {:.2f} %)'\
                      .format(self.paras.load, metric, score)
                self.verbose(msg)

    def verbose(self, msg, display_step=False):
        ''' Verbose function for print information to stdout'''
        header = '['+human_format(self.step)+']' if display_step else '[INFO]'
        if self.paras.verbose:
            if type(msg) == list:
                for m in msg:
                    print(header, m.ljust(100))
            else:
                print(header, msg.ljust(100))

    def progress(self, msg):
        ''' Verbose function for updating progress on stdout 
            Do not include newline in msg '''
        if self.paras.verbose:
            sys.stdout.write("\033[K")  # Clear line
            print('[Ep {}] {}'.format(human_format(self.cur_epoch), msg), end='\r')

    def write_log(self, log_name, log_dict, bins=None):
        ''' Write log to TensorBoard
            log_name  - <str> Name of tensorboard variable 
            log_dict - <dict>/<array> Value of variable (e.g. dict of losses)
        '''
        if log_dict is not None:
            if type(log_dict) is dict:
                log_dict = {key: val for key, val in log_dict.items() if (
                    val is not None and not math.isnan(val))}
                self.log.add_scalars(log_name, log_dict, self.step)
            elif 'Hist.' in log_name or 'Spec' in log_name:
                img, form = log_dict
                self.log.add_image(log_name,img, global_step=self.step, dataformats=form)
            else:
                raise NotImplementedError

    def save_checkpoint(self, f_name, metric, score, show_msg=True):
        '''' pt saver
            f_name - <str> the name of ckpt (w/o prefix), overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
            "epoch": self.cur_epoch,
            metric: score
        }
        torch.save(full_dict, ckpt_path)
        if show_msg:
            msg = "Saved checkpoint (epoch = {}, {} = {:.2f}) and status @ {}"
            self.verbose(msg.format(
                        human_format(self.cur_epoch), metric, score, ckpt_path))
        return ckpt_path


    # ----------------------------------- Abtract Methods ------------------- #
    @abc.abstractmethod
    def load_data(self):
        '''
        Called by main to load all data
        After this call, data related attributes should be setup 
        (e.g. self.tr_set, self.dev_set)
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def set_model(self):
        '''
        Called by main to set models
        After this call, model related attributes should be setup 
        The followings MUST be setup
            - self.model (torch.nn.Module)
            - self.optimizer (src.Optimizer),
        Loading pre-trained model should also be performed here 
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def exec(self):
        '''
        Called by main to execute training/inference
        '''
        raise NotImplementedError