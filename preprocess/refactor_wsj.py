import os
import glob
import argparse
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from subprocess import check_call, CalledProcessError


# Arguments
parser = argparse.ArgumentParser(description='Build up complete dataset.')
parser.add_argument('--wsj_root', default=None, type=str,
                    help='Path to downloaded raw WSJ', required=True)
parser.add_argument('--dest', default=None, type=str,
                    help='Path to store WSJ wav files.', required=True)
parser.add_argument('--sph2pipe_bin', default='sph2pipe_v2.5/sph2pipe', type=str,
                    help='Path to the binary file of sph2pipe', required=False)
parser.add_argument('--n_workers', default=-1, type=int,
                    help='Number of workers for multiprocessing', required=False)
parser.add_argument('--debug', action='store_true', 
                    help='Debug mode', required=False)
args = parser.parse_args()

# Convert function
def wv1_to_wav(file_path):
    wav_name = str(file_path).rsplit('/',1)[-1].replace('.wv1','.wav')
    outputpath = os.path.join(args.dest,'wav',wav_name)
    return check_call([args.sph2pipe_bin, '-f', 'rif', file_path, outputpath])

# Script begins
print('===> Listing all WSJ wv1 files...')
wav_dir = os.path.join(args.dest,'wav')
os.makedirs(wav_dir, exist_ok=True)
if 'wsj0' in args.wsj_root:
    wv1_regex = os.path.join('wsj/acoustics/cds/*/wsj*/*/*/*.wv1')
else:
    wv1_regex = os.path.join('*/*/*/*/*.wv1')
wv1_list = list(Path(args.wsj_root).glob(wv1_regex))

print('===> Converting {} wv1 files to wav files...'.format(len(wv1_list)))
assert os.path.isfile(args.sph2pipe_bin)
pool = multiprocessing.Pool(
    processes = multiprocessing.cpu_count()-1 if args.n_workers==-1 else args.n_workers)
progress = tqdm(total=len(wv1_list))
def update(*a):
    progress.update()
search_results = []
for file_path in wv1_list:
    search_results.append(pool.apply_async(wv1_to_wav, [file_path], callback=update))
pool.close()
pool.join()
progress.close()
status = [1 for r in tqdm(search_results, desc='Collecting results') if r.get()!=0]
print('===> Done ({} convertion failed)'.format(sum(status)))
print('')