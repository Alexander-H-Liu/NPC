import torch
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

MAX_SPKR_CNT = 259


class DSet(Dataset):
    ''' This is the raw WSJ parser '''
    def __init__(self, path, split):
        # Setup
        self.path = path
        self.wav_form = join(path, 'wav', '{}.wav')
        # List all wave files
        self.file_list = []
        spk2id = {}
        self.utt2id = {}
        spk_cnt = 0
        for s in split:
            with open(join(path,'spk',s), 'r') as f:
                # Notes on using setting from Yu-An
                #    All speakers
                #       - Must be sorted in same order in different split
                #       - Must appear at least once in all split
                #    Using only up to 259 spkrs
                for line in f:
                    uttr,spk = line.strip().split()
                    if spk not in spk2id:
                        spk2id[spk] = spk_cnt
                        spk_cnt +=1
                        if spk_cnt >= MAX_SPKR_CNT:
                            break
                    self.utt2id[uttr] = spk2id[spk]
        self.file_list = list(self.utt2id.keys())

    def __getitem__(self, index):
        fid = self.file_list[index]
        return self.wav_form.format(fid), self.utt2id[fid]

    def __len__(self):
        return len(self.file_list)

def collect_batch(batch, audio_transform, audio_max_frames, mode):
    '''Collects a batch, should be list of <str> file_path '''
    # Load Batch
    file_id, audio_feat, spkr_label, audio_len = [], [], [], []
    with torch.no_grad():
        for wav,spkr in batch:
            file_id.append(wav.rsplit('/',1)[-1].replace('.wav',''))
            # Audio feature (sequence) on-the-fly
            x = audio_transform(filepath=wav)
            # Crop to avoid batch too large
            if len(x)>audio_max_frames:
                x = x[:audio_max_frames]
            audio_feat.append(x)
            audio_len.append(len(x))
            spkr_label.append(spkr)
        # Descending audio length within each batch
        audio_len, audio_feat, spkr_label, file_id = zip(*[(fl, f, spk, fid)
            for fl, f, spk, fid in sorted(zip(audio_len, audio_feat, spkr_label, file_id),
            reverse=True, key=lambda x:x[0])])
        # Zero padding
        audio_feat = pad_sequence(audio_feat, batch_first=True)
        spkr_label = torch.LongTensor(spkr_label)
        return file_id, audio_feat, audio_len, spkr_label

