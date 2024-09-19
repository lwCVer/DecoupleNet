
## This repository is the official implementation of "DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for Remote Sensing Visual Tasks".
> [**DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for
Remote Sensing Visual Tasks**]  
> Wei Lu, Si-Bao Chen*, Qing-Ling Shu, Jin Tang, and Bin Luo, Senior Member, IEEE 
> 
>  *IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2024*
> 
## Introduction

The master branch is built on MMRotate which works with **PyTorch 1.6+**.

DecoupleNet backbone code is placed under mmrotate/models/backbones/, and the train/test configure files are placed under configs/lwganet/ 


## Results and models

Imagenet 300-epoch pre-trained DecoupleNet-D0 backbone: [Download](https://github.com/lwCVer/)

Imagenet 300-epoch pre-trained DecoupleNet_D1 backbone: [Download](https://github.com/lwCVer/)

Imagenet 300-epoch pre-trained DecoupleNet_D2 backbone: [Download](https://github.com/lwCVer/)

DOTA1.0

|             Model              |  mAP  | training mode | Batch Size |                                                       Configs                                                       |                                                              Download                                                               |
|:------------------------------:|:-----:|---------------|:----------:|:-------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
| DecoupleNet_D0 (1024,1024,200) | 77.38 | single-scale  |    2\*4    | [ORCNN_DecoupleNet_D0_fpn_le90_dota10_ss_e36](./configs/DecoupleNet/ORCNN_DecoupleNet_D0_fpn_le90_dota10_ss_e36.py) |          [model](https://github.com/lwCVer/)           |
| DecoupleNet_D2 (1024,1024,200) | 78.04 | single-scale  |    2\*4    | [ORCNN_DecoupleNet_D2_fpn_le90_dota10_ss_e36](./configs/DecoupleNet/ORCNN_DecoupleNet_D2_fpn_le90_dota10_ss_e36.py) |          [model](https://github.com/lwCVer/)           |


DIOR-R 

|                    Model                     |  mAP  | Batch Size |
| :------------------------------------------: |:-----:| :--------: |
|                   LWGANet_L2                   | 67.08 |    1\*8    |

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [mmrotate](https://github.com/open-mmlab/mmrotate).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n DecoupleNet-Det python=3.8 -y
conda activate DecoupleNet-Det
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
# git clone https://github.com/open-mmlab/mmrotate.git
# cd mmrotate
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation
```
@article{lu2023robust,
  title={DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for Remote Sensing Visual Tasks},
  author={Lu, Wei and Chen, Si-Bao and Shu, Qing-Ling and Tang, Jin and Luo, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={},
  pages={},
  year={2024},
  publisher={IEEE}
}
```
