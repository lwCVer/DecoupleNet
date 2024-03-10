# DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for Remote Sensing Visual Tasks

This is the official Pytorch/Pytorch implementation of the paper: <br/>
> [**DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for
Remote Sensing Visual Tasks**]  
> Wei Lu, Si-Bao Chen, Jin Tang, and Bin Luo    
> 

--- 
## Abstract

In the realm of computer vision (CV), achieving a balance between speed and accuracy remains a significant challenge. Recent efforts in this domain have focused on developing lightweight backbone networks that aim to find an optimal balance between computational efficiency and the ability to extract relevant features. However, when applied to remote sensing (RS) imagery, where the detection of small objects is often critical, these lightweight networks frequently fall short in terms of performance. To tackle these specific challenges in RS image analyses, this article introduces DecoupleNet, an innovative lightweight backbone network tailored for RS visual tasks, particularly suited for resource-constrained environments. DecoupleNet incorporates two key innovations: feature integration downsampling (FID) module and multi-branch feature decoupling (MBFD) module. FID module is designed to preserve small object features during the downsampling process, while MBFD module aims to enhance the representation of small object features and multi-scale object features via a novel feature decoupling approach. Our comprehensive evaluation across two distinct RS visual tasks demonstrates DecoupleNet’s superior ability to balance accuracy and computational efficiency compared to existing lightweight networks. Specifically, on the NWPU-RESISC45 classification dataset, DecoupleNet D0 achieves a top-1 accuracy of 95.30%, surpassing the performance of FasterNet T0 by 2%, while requiring fewer parameters and computational overheads. In object detection tasks, using the DOTA 1.0 validation set, DecoupleNet D0 records an accuracy of 69.78%, outperforming FasterNet T0 by 4.65%. Our findings open new avenues for advancing RS image analyses on resource-constrained devices, addressing a pivotal gap in the field.



<p align="center">  Illustration of DecoupleNet architecture.
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/over_view.png" width=100% 
class="center">
</p> 



<p align="center">  Illustration of various downsampling modules. (a) illustrates convolutional downsampling for selected regions, employing a 3 × 3 convolutional kernel with stride 2, and padding of 1. (b) depicts max-pooling with kernel and stride of 2. (c) provides in-depth depictions of feature integration downsampling (FID) module. Gconv, PII, MaxD, DWConvD, and Cat denote group convolution, partial information interaction, max-pooling downsampling, depthwise separable convolutional downsampling, and concatenation, respectively. (d) illustrates the detailed exhibition of PII in FID module.
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/FID.png" width=70% 
class="center">
</p>  


<p align="center">  Illustration of decouple block. Conv, MRLA, and GA denote convolution, medium-range lightweight attention, and global attention, respectively.
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/Decouple_block.png" width=70% 
class="center">
</p> 

## Image Classification
### 1. Dependency Setup
Create an new conda virtual environment
```
conda create -n DecoupleNet python=3.7 -y
conda activate DecoupleNet
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
Clone this repo and install required packages:
```
git clone https://github.com/lwCVer/DecoupleNet
cd DecoupleNet/
pip install -r requirements.txt
```

### 2. Dataset Preparation

You can download our already sliced [NWPU-RESISC45](https://github.com/lwCVer/RFD/releases/download/untagged-f0c18b912acc14db4ea2/NWPU-RESISC45.tar.xz) dataset, or download the [NWPU-RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45) classification dataset from the official document and structure the data as follows:
```
/path/to/NWPU-RESISC45/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```


### 3. Training

```
python train.py 
```
To train other models, `train.py` need to be changed.     


### 4. Experimental Results


<p align="center">Comprehensive comparison of top-1 accuracy, latency, and parameters on NWPU-RESISC45 validation set. Latency test on both NVIDIA AGXXAVIER (ARM) and Intel Core i9-11900 (CPU). The area of each circle is proportional to the number of parameters in model. DecoupleNet with red circles achieves the best balance between accuracy, latency, and parameters.
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig1.png" width=90% 
class="center">
</p>  



<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_cla.png" width=90% 
class="center">Performance of models on NWPU-RESISC45. We presented results for Type, Parameters (Params.), FLOPs, Top-1 Accuracy (Acc.), and Latency across different methods. Latency measurements were acquired through model execution on both the NVIDIA AGX-XAVIER (ARM) and Intel i9-11900 (CPU) platforms, employing batch sizes of 32 and 1 for evaluation, respectively. The symbol * indicated a training image size of 256 * 256.
</p> 

4.3 
<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_det.png" width=100% 
class="center">Object detection performance summary. The table provided object detection results for the DOTA 1.0, DIOR-R, and HRSC2016 datasets. We included backbone network parameters, backbone FLOPs, and mAP. The RoI Transformer utilized as rotation object detection head. Model trained for 12 epochs on DOTA 1.0 dataset and 36 epochs on DIOR-R and HRSC2016 datasets.
</p> 


4.4 
<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_sod.png" width=60% 
class="center">Main small object detection results % of FasterNet and DecoupleNet on DOTA1.0 validation set. SP: swimming pool. HC: helicopter. SV: small vehicle. ST: storage tank. Red and blue denoted the best and second-best performance of each column.
</p> 



### 5. Pre-trained Models (Pre-trained on RSD46) 

|     Models     | #Params. (M) | FLOPs (G) | Top-1 acc. | Latency (ms)  <br/> ARM  /  CPU |                                          Weights                                           |
|:--------------:|:------------:|:---------:|:----------:|:-------------------------------:|:------------------------------------------------------------------------------------------:|
| DecoupleNet D0 |     1.96     |   0.285   |   95.30    |            2.3 / 6.8            | [D0](https://github.com/lwCVer/DecoupleNet/releases/download/pre-train/DecoupleNet_D0.pth) |
| DecoupleNet D1 |     4.06     |   0.630   |   95.58    |           3.3 / 11.5            | [D1](https://github.com/lwCVer/DecoupleNet/releases/download/pre-train/DecoupleNet_D1.pth) |
| DecoupleNet D2 |     6.93     |   1.110   | **95.87**  |           4.3 / 15.4            | [D2](https://github.com/lwCVer/DecoupleNet/releases/download/pre-train/DecoupleNet_D2.pth) |

   



## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) and [mmdetection](https://github.com/open-mmlab/mmdetection) repositories.

If you have any questions about this work, you can contact me. Email: 2858191255@qq.com.

Your star is the power that keeps us updating github.

## Citation
If you find this repository helpful, please consider citing:
```
```
