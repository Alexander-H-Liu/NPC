import torch
from functools import partial
from src.audio import create_transform
from torch.utils.data import DataLoader


def create_dataset(name, path, batch_size, audio_max_frames,
                   train_split, dev_split, test_split=None):
    ''' Interface for creating dataset '''

    # Import dataset & collect function from target dataset
    try:
        ds = __import__(".".join(["dataset",name.lower()]),
                        fromlist=['DSet','collect_batch'])
        Dataset = ds.DSet
        collect_fn = ds.collect_batch
    except:
        raise NotImplementedError
    # Create dataset (tr/dv set should always be provided)
    tr_set = Dataset(path, train_split)
    dv_set = Dataset(path, dev_split)
    # Messages to show
    msg_list = _data_msg(name, train_split.__str__(), tr_set,
                         dev_split.__str__(), dv_set, audio_max_frames,
                         batch_size)
    # Test set of downstream task included if specified
    if test_split is None:
        tt_set = None
    else:
        tt_set = Dataset(path, test_split)
        msg_list.append('           | Test sets = {}\t| Size = {}'\
                        .format(test_split.__str__(),len(tt_set)))

    return tr_set, dv_set, tt_set, batch_size, \
           msg_list, collect_fn, audio_max_frames


def prepare_data(n_jobs, dev_n_jobs, use_gpu, pin_memory, dataset, audio):
    ''' Prepare dataloader for training/validation'''

    # Audio feature extractor
    audio_transform, audio_dim = create_transform(audio.copy())
    data_msg = audio_transform.create_msg()

    # Create dataset
    tr_set, dv_set, tt_set, batch_size, msg, collect_fn, audio_max_frames =\
        create_dataset( **dataset)
    data_msg += msg

    # Collect function
    collect_tr = partial(collect_fn, audio_max_frames=audio_max_frames,
                            audio_transform=audio_transform, mode='train')
    collect_dv = partial(collect_fn, audio_max_frames=audio_max_frames,
                            audio_transform=audio_transform, mode='dev')
    # Create data loader
    tr_set = DataLoader(tr_set, batch_size=batch_size, shuffle=True,
                        drop_last=True, collate_fn=collect_tr,
                        num_workers=n_jobs, pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=batch_size, shuffle=False,
                        drop_last=False, collate_fn=collect_dv,
                        num_workers=dev_n_jobs, pin_memory=pin_memory)

     # Prepare testset if needed
    if tt_set is not None:
        collect_tt = partial(collect_fn, audio_max_frames=audio_max_frames,
                            audio_transform=audio_transform, mode='test')
        tt_set = DataLoader(tt_set, batch_size=batch_size, shuffle=False,
                        drop_last=False, collate_fn=collect_tt,
                        num_workers=dev_n_jobs, pin_memory=pin_memory)

    return tr_set, dv_set, tt_set, audio_dim, data_msg

def _data_msg(name, tr_spt, tr_set, dv_spt, dv_set, audio_max_frames, bs):
    ''' List msg for verbose function '''
    msg_list = []
    msg_list.append('Data spec. | Dataset = {}\t| Max Frame = {}\t| Batch size = {}'\
                    .format(name, audio_max_frames, bs))
    msg_list.append('           | Train sets = {}\t| Size = {}'\
                    .format(tr_spt, len(tr_set)))
    msg_list.append('           | Dev sets = {}\t| Size = {}'\
                    .format(dv_spt, len(dv_set)))
    return msg_list