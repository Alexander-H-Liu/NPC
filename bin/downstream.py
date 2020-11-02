import os
import yaml
import copy
import torch
from src.optim import Optimizer
from src.data import prepare_data
from src.util import human_format, cal_per
from src.solver import BaseSolver # Some basic functions
from model.classifier import CLF

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self, config, paras):
        super().__init__(config, paras)
        # Logger settings
        self.best_dev_er = 1.0
        self.cur_epoch = 0
        # Configs following self-supervised learning
        self.task = self.paras.task
        assert self.task in ['phn-clf','spk-clf'], 'unsupported task'
        self.ssl_config = yaml.load(
                            open(self.config['model']['feat']['config'], 'r'),
                            Loader=yaml.FullLoader)
        self.feature = self.ssl_config['model']['method']
        if self.feature == 'npc' and 'spec' in self.config['model']['feat']:
            # NPC has additional option to use unmasked feature
            self.feat_spec = self.config['model']['feat']['spec']
        else:
            self.feat_spec = None
        self.config['data']['audio'] = self.ssl_config['data']['audio']

    def fetch_data(self, data, train=True):
        ''' Move data to device '''
        file_id, audio_feat, audio_len, label = data
        if self.gpu:
            audio_feat = audio_feat.cuda()
            label = label.cuda()
        # Extract feature
        with torch.no_grad():
            if self.feat_spec is not None:
                # Get unmasked feature from particular NPC layer
                n_layer_feat = int(self.feat_spec.split('-')[-1])
                audio_feat = self.feat_extractor.get_unmasked_feat(audio_feat,n_layer_feat)
            elif self.feature == 'npc':
                # Get masked feature from NPC
                _, audio_feat = self.feat_extractor(audio_feat,testing=True)
            else:
                # Get feature from APC based model
                _, audio_feat = self.feat_extractor(audio_feat, audio_len,
                                                    testing=True)
            # Mean pool feature for spkr classification
            if self.task == 'spk-clf':
                single_feat = []
                for a_feat, a_len in zip(audio_feat,audio_len):
                    single_feat.append(a_feat[:a_len].mean(dim=0))
                audio_feat = torch.stack(single_feat, dim=0)
        return file_id, audio_feat, audio_len, label

    def load_data(self):
        ''' Load data for training/validation '''
        self.tr_set, self.dv_set, self.tt_set, self.audio_dim, msg = \
            prepare_data(self.paras.njobs,self.paras.dev_njobs,self.paras.gpu,
                         self.paras.pin_memory, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup model and optimizer '''
        # Load SSL models for feature extraction
        self.verbose([' Load feat. extractor ckpt from '\
                        +self.config['model']['feat']['ckpt']])
        if self.feature in ['apc','vqapc']:
            from model.apc import APC as Net
        elif self.feature == 'npc':
            from model.npc import NPC as Net
            if self.feat_spec is not None:
                self.verbose([' Using specific feature: '+self.feat_spec])
        else:
            raise NotImplementedError
        self.feat_extractor = Net(input_size=self.audio_dim,
                                  **self.ssl_config['model']['paras'])
        ckpt = torch.load( self.config['model']['feat']['ckpt'],
                map_location=self.device if self.mode == 'train' else 'cpu')
        ckpt['model'] = {k.replace('module.','',1):v \
                            for k,v in ckpt['model'].items()}
        self.feat_extractor.load_state_dict(ckpt['model'])

        # Classifier model
        self.model = CLF(feat_dim=self.feat_extractor.code_dim,
                         **self.config['model']['clf'])
        if self.gpu:
            self.feat_extractor = self.feat_extractor.cuda()
            self.feat_extractor.eval()
            self.model = self.model.cuda()
        model_paras = [{'params': self.model.parameters()}]

        # Losses
        ignore_idx = 0 if self.task == 'phn-clf' else -1
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)
        if self.gpu:
            self.loss = self.loss.cuda()

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        self.load_ckpt()
        self.model.train()


    def exec(self):
        ''' Training End-to-end ASR system '''
        if self.paras.mode =='train':
            self.verbose('Total training epoch {}.'.format(
                human_format(self.epoch)))
            self.timer.set()
            ep_len = len(self.tr_set)
            for ep in range(self.epoch):
                if ep>0:
                    # Lr decay if needed
                    self.optimizer.decay()
                for data in self.tr_set:
                    # Pre-step :  do zero_grad
                    self.optimizer.pre_step(self.step)
                    
                    # Fetch data
                    self.timer.cnt('rd')
                    _, audio_feat, audio_len, label = self.fetch_data(data)

                    # Forward
                    pred = self.model(audio_feat)
                    if self.task =='phn-clf':
                        pred = pred.permute(0,2,1) # BxCxT for phn clf
                    loss = self.loss(pred, label)
                    self.timer.cnt('fw')

                    # Backprop
                    grad_norm = self.backward(loss)
                    self.step += 1

                    # Logger
                    if (self.step == 1) or (self.step%self.PROGRESS_STEP == 0):
                        self.progress(' {:2.1f} % | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                                  .format(100*float(self.step%ep_len)/ep_len,
                                          loss.cpu().item(),
                                          grad_norm,
                                          self.timer.show()))
                        self.write_log(self.task+'_loss', {'tr': loss})
                        if self.task == 'phn-clf':
                            tr_er = cal_per(pred,label,audio_len)[0]
                        else:
                            tr_er = (pred.argmax(dim=-1)!=label)
                            tr_er = tr_er.sum().detach().cpu().float()/len(label)
                        self.write_log(self.task+'_er',{'tr':tr_er})
                    # End of step
                    self.timer.set()
                # End of epoch
                self.cur_epoch += 1
                self.validate()

        # Test at the end
        self.validate(test=True)
        self.log.close()

    def validate(self, test=False):
        # Eval mode
        self.model.eval()
        val_loss = []
        split = 'dev'
        val_hit,val_total = 0.0, 0.0
        ds = self.tt_set if test else self.dv_set

        # In training mode, best model is stored in RAM for test
        # ToDo: load ckpt
        if test:
            split = 'test'
            if self.paras.mode =='train':
                self.model = self.best_model
                if self.gpu:
                    self.model = self.model.cuda()

        for i, data in enumerate(ds):
            self.progress('Valid step - {}/{}'.format(i+1, len(ds)))
            # Fetch data
            _, audio_feat, audio_len, label = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                # Prediction
                pred = self.model(audio_feat)
                if self.task == 'phn-clf':
                    pred = pred.permute(0,2,1) # BxCxT
                # Accumulate batch result
                val_loss.append(self.loss(pred, label))
                if self.task == 'phn-clf':
                    _, hit, total = cal_per(pred, label, audio_len)
                    val_hit += hit
                    val_total += total
                else:
                    hit = (pred.argmax(dim=-1)==label).sum()
                    val_hit += hit.detach().cpu().float()
                    val_total += len(label)
                # Write testing prediction if needed
                if test and self.paras.write_test:
                    if self.task == 'phn-clf':
                        pred = pred.argmax(dim=1).detach().cpu()
                    label = label.cpu()
                    with open(os.path.join(self.ckpdir,self.task+'.csv'),'a') as f:
                        for p,l,a_len in zip(pred,label,audio_len):
                            for x, y in zip(p[:a_len].tolist(),l[:a_len].tolist()):
                                f.write('{}\t{}\n'.format(x,y))

        # Record metric, store ckpt by dev error rate
        val_loss = sum(val_loss)/len(val_loss)
        val_er = 1.0-val_hit/val_total
        self.write_log(self.task+'_loss', {split:val_loss})
        self.write_log(self.task+'_er', {split:val_er})
        if split=='dev' and self.best_dev_er > val_er:
            self.best_dev_er = val_er
            self.save_checkpoint('best.pth',self.task+'_er',val_er)
            self.best_model = copy.deepcopy(self.model.cpu()) # Clone for test
        
        # Resume training
        if self.gpu:
            self.model = self.model.cuda()
        self.model.train()