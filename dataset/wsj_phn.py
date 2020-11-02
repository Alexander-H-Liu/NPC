import torch
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DSet(Dataset):
    ''' This is the WSJ parser '''
    def __init__(self, path, split):
        # Setup
        self.path = path
        self.wav_form = join(path, 'wav', '{}.wav')
        self.phn_form = join(path, 'phn', '{}.pt')
        # List all wave files
        self.file_list = []
        for s in split:
            s_list = pd.read_csv(join(path,'meta',s+'_phn.csv'),header=None)[0].tolist()
            assert len(s_list) > 0, "No data found @ {}".format(join(path,s))
            self.file_list += s_list

    def __getitem__(self, index):
        fid = self.file_list[index]
        return self.wav_form.format(fid), self.phn_form.format(fid)

    def __len__(self):
        return len(self.file_list)

def collect_batch(batch, audio_transform, audio_max_frames, mode):
    '''Collects a batch, should be list of <str> file_path '''    
    # Load Batch
    file_id, audio_feat, phn_seq, audio_len = [], [], [], []
    with torch.no_grad():
        for wav,phn in batch:
            file_id.append(wav.rsplit('/',1)[-1].replace('.wav',''))
            # Audio feature (sequence) on-the-fly
            x = audio_transform(filepath=wav)
            # Phn label sequence (test set shouldn't be cropped)
            if mode =='test':
                phn = phn.replace('.pt','_nocrop.pt')
            y = torch.load(phn)+1 # 0 = pad
            # Crop to avoid batch too large
            x,y = _crop(x,y,audio_max_frames, mode)
            audio_feat.append(x)
            audio_len.append(len(x))
            phn_seq.append(y[:len(x)])
        # Descending audio length within each batch
        audio_len, audio_feat, phn_seq, file_id = zip(*[(fl, f, phn, fid)
            for fl, f, phn, fid in sorted(zip(audio_len, audio_feat, phn_seq, file_id),
            reverse=True, key=lambda x:x[0])])
        # Zero padding
        audio_feat = pad_sequence(audio_feat, batch_first=True)
        phn_seq = pad_sequence(phn_seq, batch_first=True)
        return file_id, audio_feat, audio_len, phn_seq

def _crop(x, y, max_len, mode):
    if len(x)>len(y):
        if mode == 'test':
            raise NotImplementedError('Test set are not supposed to be cropped')
        else:
            # Crop files that are too long
            x = x[:len(y)]
    if len(x) > max_len:
        return x[:max_len],y[:max_len]
    else:
        return x,y