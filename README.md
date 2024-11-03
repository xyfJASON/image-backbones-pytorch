# Image-Backbones-Implementations

Implement image backbones with PyTorch.

<br/>



## Installation

> The code is tested with python 3.12, torch 2.4.1 and cuda 12.4.

Clone this repo:
```
git clone https://github.com/xyfJASON/image-backbones-pytorch.git
cd image-backbones-pytorch
```

Create and activate a conda environment:

```shell
conda create -n backbones python=3.12
conda activate backbones
```

Install dependencies:

```shell
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

<br/>



## Training

```shell
accelerate-launch main.py [-c CONFIG] [-e EXP_DIR] [--xxx.yyy zzz ...]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Results (logs, checkpoints, tensorboard, etc.) of each run will be saved to `EXP_DIR`. If `EXP_DIR` is not specified, they will be saved to `runs/exp-{current time}/`.
- To modify some configuration items without creating a new configuration file, you can pass `--key value` pairs to the script. For example, the default optimizer in `./configs/resnet18_cifar10.yaml` is SGD, and if you want to change it to Adam without bothering to create a new file, you can simply pass `--train.optim.type Adam`.

For example, to train resnet18 on CIFAR-10:

```shell
accelerate-launch main.py -c ./configs/resnet18_cifar10.yaml
```

<br/>



## Results

### CIFAR-10 Benchmark

<table>
  <tr>
    <th align="center">models</th>
    <th align="center">#params</th>
    <th align="center">MACs</th>
    <th align="center">acc@1(%)</th>
  </tr>
  <tr>
    <td align="center">VGG-11</td>
    <td align="center">9.20M</td>
    <td align="center">0.15G</td>
    <td align="center">90.97</td>
  </tr>
  <tr>
    <td align="center">VGG-19 (BN)</td>
    <td align="center">20.0M</td>
    <td align="center">0.40G</td>
    <td align="center">94.00</td>
  </tr>
  <tr>
    <td align="center">ResNet-18</td>
    <td align="center">11.2M</td>
    <td align="center">5.59G</td>
    <td align="center">95.64</td>
  </tr>
  <tr>
    <td align="center">PreActResNet-18</td>
    <td align="center">11.2M</td>
    <td align="center">5.59G</td>
    <td align="center">95.45</td>
  </tr>
  <tr>
    <td align="center">ResNeXt-29 (32x4d)</td>
    <td align="center">4.78M</td>
    <td align="center">6.90G</td>
    <td align="center">95.16</td>
  </tr>
  <tr>
    <td align="center">SE-ResNet-18</td>
    <td align="center">11.3M</td>
    <td align="center">5.59G</td>
    <td align="center">95.65</td>
  </tr>
  <tr>
    <td align="center">CBAM-ResNet-18</td>
    <td align="center">11.3M</td>
    <td align="center">5.59G</td>
    <td align="center">95.49</td>
  </tr>
  <tr>
    <td align="center">MobileNet</td>
    <td align="center">3.22M</td>
    <td align="center">0.48G</td>
    <td align="center">92.09</td>
  </tr>
  <tr>
    <td align="center">ShuffleNet 1x (g=8)</td>
    <td align="center">0.91M</td>
    <td align="center">0.50G</td>
    <td align="center">92.82</td>
  </tr>
  <tr>
    <td align="center">ViT-Tiny/4</td>
    <td align="center">5.36M</td>
    <td align="center">0.37G</td>
    <td align="center">85.66</td>
  </tr>
</table>

Note: MACs are calculated by [fvcore](https://github.com/facebookresearch/fvcore) library.

All the ConvNets are trained with the following settings:

- training duration: 64k steps
- batch size: 256
- learning rate: start with 0.1, end with 0.001 using a cosine annealing scheduler, no warm-up
- optimizer: SGD, weight decay 5e-4, momentum 0.9

The ViTs are trained with the following settings:

- training duration: 64k steps
- batch size: 512
- learning rate: start with 0.001, end with 0.00001 using a cosine annealing scheduler, no warm-up
- optimizer: Adam, weight decay 5e-5, betas (0.9, 0.999)

