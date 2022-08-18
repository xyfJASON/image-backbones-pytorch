# Image Classification

Reproduce image classification models with PyTorch.



## Training Process

1. Modify the configuration file `config.yml`
   - Choose `model`, `optimizer`, `scheduler`, etc.
   - Set hyperparameters such as `batch_size`, `lr`, `epochs`, etc.
   - Other settings such as `save_freq`, `resume`, etc.

2. Start training

   For single GPU / CPU, run command:

   ```shell
   python train.py
   ```

   For multiple GPUs (e.g. 2 GPUs), run command:

   ```shell
   torchrun --nproc_per_node 2 train.py
   ```



## Results

<table style="text-align:center">
<tr>
    <th rowspan="2">Models</th>
    <th rowspan="2">#Params(million)</th>
    <th colspan="2">Acc.(%)</th>
</tr>
<tr>
    <th>CIFAR10</th>
    <th>CIFAR100</th>
</tr>
<tr>
	<td>vgg11</td><td>9.20</td><td>91.17</td><td>62.97</td>
</tr>
<tr>
	<td>vgg19 (bn)</td><td>20.0</td><td>93.21</td><td>73.22</td>
</tr>
<tr>
    <td>resnet18</td><td>11.2</td><td>95.27</td><td>77.74</td>
</tr>
<tr>
    <td>preactresnet18</td><td>11.2</td><td>95.30</td><td>77.34</td>
</tr>
<tr>
    <td>resnext29 (32x4d)</td><td>4.78</td><td>94.68</td><td>78.62</td>
</tr>
<tr>
    <td>se-resnet18</td><td>11.3</td><td>95.08</td><td>78.06</td>
</tr>
<tr>
    <td>cbam-resnet18</td><td>11.3</td><td>95.27</td><td>77.86</td>
</tr>
<tr>
	<td>mobilenet</td><td>3.22</td><td>92.34</td><td>70.70</td>
</tr>
<tr>
	<td>shufflenet 1x (g=8)</td><td>0.91</td><td>92.18</td><td>71.83</td>
</tr>
<tr>
	<td>vit-tiny-patch4-32</td><td>5.36</td><td>85.83</td><td>56.98</td>
</tr>
<tr>
	<td>vit-small-patch4-32</td><td>21.34</td><td>85.79</td><td>58.31</td>
</tr>
</table>
