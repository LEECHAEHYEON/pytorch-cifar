# Pytorch Baseline

Original : https://github.com/kuangliu/pytorch-cifar by kuangliu.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Log and checkpoint
Add logger and checkpoint.
Log and checkpoint files will be saved into each directory.
(./log/[cur_time] && ./checkpoint/[cur_time])

## Learning rate adjustment
You can adjust learning rate with manual method or auto method.
Train with auto scheduler with `python3 main.py --scheduler`
or you can use a manual way with adjust_learning_rate().

With `python3 main.py --help`, you will get more information.

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

