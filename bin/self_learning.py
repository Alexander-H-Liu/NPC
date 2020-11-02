import os
import torch
from src.optim import Optimizer
from src.data import prepare_data
from src.util import human_format, draw
from src.solver import BaseSolver # Some basic functions

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self, config, paras):
        super().__init__(config, paras)
        # Logger settings
        self.val_loss = 1000
        self.cur_epoch = 0

    def fetch_data(self, data):
        ''' Move data to device '''
        file_id, audio_feat, audio_len = data
        if self.gpu:
            audio_feat = audio_feat.cuda()
        return file_id, audio_feat, audio_len

    def load_data(self):
        ''' Load data for training/validation '''
        self.tr_set, self.dv_set, _, self.audio_dim, msg = \
            prepare_data(self.paras.njobs, self.paras.dev_njobs, self.paras.gpu,
                         self.paras.pin_memory, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup model and optimizer '''
        # Model
        self.method = self.config['model']['method']
        if self.method in ['apc','vqapc']:
            self.n_future = self.config['model']['n_future']
            from model.apc import APC as Net
        elif self.method == 'npc':
            from model.npc import NPC as Net
        else:
            raise NotImplementedError
        self.model = Net(input_size=self.audio_dim, **self.config['model']['paras'])
        if self.gpu:
            self.model = self.model.cuda()
        self.verbose(self.model.create_msg())
        model_paras = [{'params': self.model.parameters()}]

        # Loss
        if 'npc' in self.method:
            # Avoid reduction for NPC for zero-padding
            self.loss = torch.nn.L1Loss(reduction='none')
        else:
            # APC family have zero-padding with torch API
            self.loss = torch.nn.L1Loss()
        if self.gpu:
            self.loss = self.loss.cuda()

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

        # ToDo:  Data Parallel?
        # self.model = torch.nn.DataParallel(self.model)
        self.model.train()

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training epoch {}.'.format(
            human_format(self.epoch)))
        self.timer.set()
        aug_loss = None
        ep_len = len(self.tr_set)

        for ep in range(self.epoch):
            # Pre-step, decay
            if ep>0:
                self.optimizer.decay()

            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                self.optimizer.pre_step(self.step)
                
                # Fetch data
                _, audio_feat, audio_len = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward real data
                if 'npc' in self.method:
                    # NPC: input = target
                    pred, _ = self.model(audio_feat)
                    loss = self.loss(pred, audio_feat)
                    # Compute loss on valid part only
                    effective_loss = 0
                    for i,a_len in enumerate(audio_len):
                        effective_loss += loss[i,:a_len,:].mean(dim=-1).sum()
                    loss = effective_loss/sum(audio_len)
                else:
                    # APC: input = shifted target
                    audio_len = [l-self.n_future for l in audio_len]
                    pred, _ = self.model(audio_feat[:,:-self.n_future,:], audio_len, testing=False)
                    loss = self.loss(pred, audio_feat[:,self.n_future:,:])
                self.timer.cnt('fw')
                # Backprop
                grad_norm = self.backward(loss)
                self.step += 1

                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    self.progress(' {:2.1f} % | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                                  .format(100*float(self.step%ep_len)/ep_len,
                                          loss.cpu().item(),
                                          grad_norm,
                                          self.timer.show()))
                    self.write_log('loss', {'tr': loss})
                    
                if (self.step == 1) or (self.step % self.PLOT_STEP == 0):
                    # Perplexity of P(token)
                    g1_ppx, g2_ppx = self.model.report_ppx()     
                    self.write_log('ppx', {'group 1':g1_ppx,
                                           'group 2':g2_ppx})
                    g1_usg, g2_usg = self.model.report_usg() # Empty cache
                    # Plots
                    if self.paras.draw:
                        g1_hist = draw(g1_usg, hist=True)
                        g2_hist = draw(g2_usg, hist=True)
                        self.write_log('VQ Group 1 Hist.',g1_hist)
                        self.write_log('VQ Group 2 Hist.',g2_hist)
                        # Some spectrograms
                        plt_idx = 0
                        self.write_log('Spectrogram (raw)', draw(audio_feat[plt_idx]))
                        self.write_log('Spectrogram (pred)', draw(pred[plt_idx]))

                # End of step
                self.timer.set()
            # End of epoch
            self.cur_epoch += 1
            self.validate()
        self.log.close()

    def validate(self):
        # Eval mode
        self.model.eval()
        dev_loss = []
        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dv_set)))
            # Fetch data
            _, audio_feat, audio_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                if 'npc' in self.method:
                    pred, _ = self.model(audio_feat, testing=True)
                    loss = self.loss(pred, audio_feat)
                    # Compute loss on valid part only
                    effective_loss = 0
                    for i,a_len in enumerate(audio_len):
                        effective_loss += loss[i,:a_len,:].mean(dim=-1).sum()
                    loss = effective_loss/sum(audio_len)
                else:
                    audio_len = [l-self.n_future for l in audio_len]
                    pred, _ = self.model(audio_feat[:,:-self.n_future,:], audio_len, testing=True)
                    loss = self.loss(pred, audio_feat[:,self.n_future:,:])
                dev_loss.append(loss.cpu().item())

        # Record metric
        dev_loss = sum(dev_loss)/len(dev_loss)
        self.write_log('loss', {'dev':dev_loss})
        if dev_loss < self.val_loss:
            self.val_loss = dev_loss
            self.save_checkpoint('best_loss.pth', 'loss', dev_loss)
        # Resume training
        self.model.train()