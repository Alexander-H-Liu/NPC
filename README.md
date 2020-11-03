# Non-Autoregressive Predictive Coding

This repository contains the implementation of Non-Autoregressive Predictive Coding (NPC) as described in [the preprint paper](https://arxiv.org/abs/2011.00406) submitted to ICASSP 2021.


A quick example for training NPC
```
python main.py --config config/self_supervised/npc_example.yml \
               --task self-learning
```

- For more complete examples including downstream tasks, please see [the example script](eg.sh).

- For preparing data, please visit [preprocess](preprocess/).

- For detailed hyperparameters setting and description, please checkout [example config file of NPC](config/self_supervised/npc_example.yml). 

- For all run-time options, use `-h` flag.

- Implementation of [Autoregressive Predictive Coding](https://arxiv.org/abs/1910.12607) (APC, 2019, Chung *et al*.) and [Vector-Quantized APC](https://arxiv.org/abs/2005.08392) (VQ-APC, 2020, Chung *et al*.) are also available using similar training/downstream execution with example config files [here](config/self_supervised/vqapc_example.yml).

## Some notes

- We found the unmasked feature produced by the last ConvBlock layer a better representation. In the phone classification tasks, switching to the unmasked feature (PER 25.6%) provided a 1.6% improvement over the masked feature (PER 27.2%). Currently, this is not included in the preprint version and will be updated to the paper in the future. Please refer to [downstream examples](config/downstream) to activate this option.

- APC/VQ-APC are implemented with the following modifications for improvement (for the unmodified version, please visit the official implementation of [APC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding) / [VQAPC](https://github.com/iamyuanchung/VQ-APC/tree/96230cc358b174b736b4c0e7664b3e72b304d9b0))

    - Multi-group VQ available for VQ-APC, but with VQ on last layer only

    - Using utterance-wised CMVN surface feature（just as NPC did)

    - Using Gumbel Softmax from [official API](https://pytorch.org/docs/stable/nn.functional.html#gumbel-softmax) of pytorch

- See [package requirement](requirements.txt) for toolkits used, `tensorboard` can be used to access log files in `--logdir`.


## Contact

Feel free to contact me for questions or feedbacks, my email can be found in the paper or my [personal page](https://alexander-h-liu.github.io).

## Citation

If you find our work and/or this repository helpful, please do consider citing us

```
@article{liu2020nonautoregressive,
  title   = {Non-Autoregressive Predictive Coding for Learning Speech Representations from Local Dependencies},
  author  = {Liu, Alexander and Chung, Yu-An and Glass, James},
  journal = {arXiv preprint arXiv:2011.00406},
  year    = {2020}
}
```
