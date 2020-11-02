import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Timer():
    ''' Timer for recording training time distribution. '''

    def __init__(self):
        self.prev_t = time.time()
        self.clear()

    def set(self):
        self.prev_t = time.time()

    def cnt(self, mode):
        self.time_table[mode] += time.time()-self.prev_t
        self.set()
        if mode == 'bw':
            self.click += 1

    def show(self):
        total_time = sum(self.time_table.values())
        self.time_table['avg'] = total_time/self.click
        self.time_table['rd'] = 100*self.time_table['rd']/total_time
        self.time_table['fw'] = 100*self.time_table['fw']/total_time
        self.time_table['bw'] = 100*self.time_table['bw']/total_time
        msg = '{avg:.3f} sec/step (rd {rd:.1f}% | fw {fw:.1f}% | bw {bw:.1f}%)\
              '.format(**self.time_table)
        self.clear()
        return msg

    def clear(self):
        self.time_table = {'rd': 0, 'fw': 0, 'bw': 0}
        self.click = 0
 

def human_format(num):
    ''' Convert number to human readable format 
        Reference :
            https://stackoverflow.com/questions/579310/\
            formatting-long-numbers-as-strings-in-python'''
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3.1f}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])

def draw(data, hist=False):
    if data is None:
        return None
    if hist:
        data = _save_canvas( data, hist)
    else:
        data = _save_canvas(data.detach().cpu().numpy().T, hist)
    return torch.FloatTensor(data),"HWC"

def _save_canvas(data, hist):
    fig, ax = plt.subplots(figsize=(20, 8))
    if not hist:
        ax.imshow(data, aspect="auto", origin="lower")
    else:
        # Bar for distribution
        ax.bar(list(range(len(data))),data)
    fig.canvas.draw()
    # Note : torch tb add_image takes color as [0,1]
    data = np.array(fig.canvas.renderer._renderer)[:,:,:-1]/255.0 
    plt.close(fig)
    return data

def cal_per(pred, label, seq_len):
    # BxCxT -> BxT
    pred = pred.argmax(dim=1).detach().cpu()
    label = label.cpu()
    hit, total = 0,0
    for idx,l in enumerate(seq_len):
        hit += sum(pred[idx,:l] == label[idx,:l])
        total += l
    return 1-(float(hit)/float(total)), hit, total

