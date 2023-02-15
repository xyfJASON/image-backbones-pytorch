# Image-Backbones-Implementations

My implementations of image backbones with PyTorch.

<br/>



## Training

```shell
python main.py [-n NAME] [-c FILE] [--opts KEY1 VALUE1 KEY2 VALUE2]
```

- To train on multiple GPUs, replace `python` with `torchrun --nproc_per_node NUM_GPUS`.
- An experiment directory will be created under `./runs/` for each run, which is named after `NAME`, or the current time if `NAME` is not specified. The directory contains logs, checkpoints, tensorboard, etc.

For example, to train resnet18 on CIFAR-10:

```shell
python main.py -c ./configs/resnet18_cifar10.yaml
```

<br/>



## Results



### CIFAR-10

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

