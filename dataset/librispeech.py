import torch
import numpy as np
from os.path import join
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DSet(Dataset):
    '''  LibriSpeech parser which takes raw LibriSpeech structure'''
    def __init__(self, path, split):
        # Setup
        self.path = path
        # List all wave files
        self.file_list = []
        for s in split:
            split_list = list(Path(join(path, s)).rglob("*.flac"))
            assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
            self.file_list += split_list

    def __getitem__(self, index):
        return self.file_list[index]

    def __len__(self):
        return len(self.file_list)

def collect_batch(batch, audio_transform, audio_max_frames, mode):
    '''Collects a batch, should be list of <str> file_path '''
    # Load Batch
    file_id, audio_feat, audio_len = [], [], []
    with torch.no_grad():
        # Load each uttr. in batch
        for f_path in batch:
            file_id.append(f_path)
            # Audio feature (sequence) on-the-fly
            y = audio_transform(filepath=f_path)
            if mode=='train':
                # Crop to avoid OOM
                y = _crop(y,audio_max_frames)
            audio_feat.append(y)
            audio_len.append(len(y))
        # Descending audio length within each batch
        audio_len, file_id, audio_feat = zip(*[(aud_l, f_id, feat)
            for aud_l, f_id, feat in sorted(zip(audio_len, file_id, audio_feat),
                                            reverse=True, key=lambda x:x[0])])
        # Zero padding
        audio_feat = pad_sequence(audio_feat, batch_first=True)
    return file_id, audio_feat, audio_len

def _crop(y,max_len):
    if len(y) > max_len:
        offset = np.random.randint(len(y)-max_len)
        return y[offset:offset+max_len]
    else:
        return y