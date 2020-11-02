## Preparing LibriSpeech

Simply download LibriSpeech from [OpenSLR](http://www.openslr.org/12/) and unzip it. Fill in the `path` in config file for self-supervised learning with the path to unzipped LibriSpeech.

## Preparing WSJ

0. Download WSJ dataset (requires [LDC](https://ldc.upenn.edu) license)

1. Download and compile [sph2pipe_v2.5](https://www.openslr.org/3/) to read WSJ dataset

```
wget http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
tar xzf sph2pipe_v2.5.tar.gz
cd sph2pipe_v2.5; gcc -o sph2pipe *.c -lm
```

2. Refactor (generate wav files and place them all together) WSJ with

```
python refactor_wsj.py --wsj_root /path/to/downloaded/wsj/ \
                       --dest /path/to/store/new/wsj/
```

4. (For phone classification only.) For each utterance, please use Kaldi to obtain force aligment and store the corresponding phone [index](phone.txt) sequence with `torch.save` at `/path/to/store/new/wsj/phn/fileid.pt` (or `fileid_nocrop.pt` for `dev93` split) where `fileid.wav` can be found at `/path/to/store/new/wsj/wav/` after previous step. Last, copy the list of `fileid` of different splits to he refactored wsj dataset for use with

```
cp -r phn_split/ /path/to/store/new/wsj/wav/meta/
```

5. (For speaker classification only.) The list of `fileid` & `speaker` pairs used in different splits are stored at `spk/`. Copy them to the refactored wsj dataset for use with

```
cp -r spk_split/ /path/to/store/new/wsj/wav/spk/
```

6. Modify the `path` in config file for downstream tasks to `/path/to/store/new/wsj/`