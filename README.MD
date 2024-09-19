# DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for Remote Sensing Visual Tasks [TGRS 2024]

This is the official Pytorch/Pytorch implementation of the paper: <br/>
> [**DecoupleNet: A Lightweight Backbone Network with Efficient Feature Decoupling for
Remote Sensing Visual Tasks**]  
> Wei Lu, Si-Bao Chen*, Qing-Ling Shu, Jin Tang, and Bin Luo, Senior Member, IEEE 
> 
>  *IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2024*
> 

--- 
## Abstract

In the realm of computer vision (CV), balancing speed and accuracy remains a 
significant challenge. Recent efforts have focused on developing lightweight 
networks that optimize computational efficiency and feature extraction. 
However, in remote sensing (RS) imagery, where small and multi-scale object detection 
is critical, these networks often fall short in performance. To address these 
challenges, DecoupleNet is proposed, an innovative lightweight backbone network 
specifically designed for RS visual tasks in resource-constrained environments. 
DecoupleNet incorporates two key modules: the feature integration downsampling (FID) 
module and the multi-branch feature decoupling (MBFD) module. FID module preserves 
small object features during downsampling, while MBFD module enhances small and 
multi-scale object feature representation through a novel decoupling approach. 
Comprehensive evaluations on three RS visual tasks demonstrate DecoupleNet's 
superior balance of accuracy and computational efficiency compared to existing 
lightweight networks. On the NWPU-RESISC45 classification dataset, DecoupleNet 
achieves a top-1 accuracy of 95.30\%, surpassing FasterNet by 2\%, with fewer 
parameters and lower computational overhead. In object detection tasks using the 
DOTA 1.0 test set, DecoupleNet records an accuracy of 78.04\%, outperforming 
ARC-R50 by 0.69\%. For semantic segmentation on the LoveDA test set, DecoupleNet 
achieves 53.1\% accuracy, surpassing UnetFormer by 0.70\%. These findings open 
new avenues for advancing RS image analysis on resource-constrained devices, 
addressing a pivotal gap in the field.

----

<p align="center"> 
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/over_view.png" width=100% 
class="center">
<p align="center">  Illustration of DecoupleNet architecture.
</p> 

----

<p align="center"> 
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/FID.png" width=70% 
class="center">
<p align="center">  
 Illustration of various downsampling modules. (a) Convolutional
downsampling for selected regions, employing a 3×3 convolutional kernel
with stride 2 and padding of 1. (b) Max-pooling with kernel and stride of 2.
(c) Detailed depiction of the feature integration downsampling (FID) module.
Gconv, PII, MaxD, DWConvD, and Cat denote group convolution, partial
information interaction, max-pooling downsampling, depthwise separable
convolutional downsampling, and concatenation, respectively.
</p>  

----

<p align="center"> 
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/Decouple_Block.png" width=70% 
class="center">
<p align="center">  Illustration of decouple block. Conv, MRLA, and GA denote 
convolution, medium-range lightweight attention, and global attention, respectively.
</p>

----

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

----


<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_cla.png" width=90% 
class="center">
<p align="center">
Performance of models on classification datasets. Red and blue denoted the best and 
second-best performance of each column. We presented results for Type, Parameters 
(Params.), FLOPs, Top-1 Accuracy (Acc.), and throughput across different methods. 
Throughput measurements were acquired through model execution on both NVIDIA GeForce
RTX 3090Ti (GPU), NVIDIA AGX-XAVIER (ARM), and Intel i9-11900K (CPU) platforms, 
employing batch sizes of 256, 32 and 16 for evaluation, respectively. The 
symbol '*' indicated a training image size of 256x256.
</p> 


----


<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig1.png" width=90% 
class="center">
<p align="center">
 Comprehensive comparison of top-1 accuracy, throughput, and parameters on 
NWPU-RESISC45 validation set. Throughput test on both NVIDIA RTX 3090Ti (GPU), 
NVIDIA AGX-XAVIER (edge device with ARM architecture), and Intel Core i9-11900K (CPU).
The area of each circle is proportional to the number of parameters in the model. 
DecoupleNet, represented by red circles, achieves the best balance between accuracy, 
throughput, and parameters.
</p>  


----


## Object Detection

<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_det_dota10.png" width=100% 
class="center">
<p align="center">
Detection performance comparison on DOTA 1.0 test set. Oriented R-CNN was utilized as 
the decoder for DecoupleNet. Red and blue denoted the best and second-best performance
of each column. *E-FormerV2 (EfficientFormerV2). LV: large vehicle. SP: swimming pool.
HC: helicopter. BR: bridge. PL: plane. SH: ship. SBF: soccer-ball field. BC: 
basketball court. GTF: ground track field. SV: small vehicle. BD: baseball diamond.
TC: tennis court. RA: roundabout. ST: storage tank. HA: harbor.
</p> 


<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_det_diorr.png" width=100% 
class="center">
<p align="center">
Experimental results on DIOR-R test set. Complexities were tested by 800x800.
</p> 


<p align="center">
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/det_vis.png" width=100% 
class="center">
<p align="center">
Visual results on DOTA 1.0 test sets. Oriented R-CNN was used as detection head. 
The images at the top demonstrated the superiority of DecoupleNet in detecting small
objects, while the traditional lightweight backbone was inadequate well in this task.
The images at the bottom showed that detailed features were essential for accurate 
detection in scenarios with occlusion, which DecoupleNet provided a unique advantage.
</p> 


----

## Semantic Segmentation


<p align="center"> 
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_seg_loveda.png" width=60% 
class="center"> 
<p align="center"> 
Segmentation experimental results on LoveDA test set with SOTA models.
</p> 


<p align="center"> 
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/fig_seg_uavid.png" width=60% 
class="center"> 
<p align="center"> 
Segmentation experimental results on UAVid test set with lightweight SOTA. 
Complexities are tested by 1024x1024 input.
</p> 


<p align="center"> 
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/seg_vis_uavid.png" width=60% 
class="center"> 
<p align="center"> 
Visual results on UAVid test set. UnetFormer was used as decoder.
</p> 

<p align="center"> 
<img src="https://github.com/lwCVer/DecoupleNet/blob/main/figures/seg_vis_loveda.png" width=60% 
class="center"> 
<p align="center"> 
Visual results on LoveDA test sets. UnetFormer was used as decoder. 
The proposed backbone, DecoupleNet, achieves optimal performance.
</p>

----


### 5. Pre-trained Models (Pre-trained on ImageNet1k) 

|     Models     | #Params. (M) | FLOPs (G) |                                          Weights                                           |
|:--------------:|:------------:|:---------:|:------------------------------------------------------------------------------------------:|
| DecoupleNet D0 |     1.96     |   0.285   | [D0](https://github.com/lwCVer/DecoupleNet/releases/download/weights/DecoupleNet_D0.pth) |
| DecoupleNet D1 |     4.06     |   0.630   | [D1](https://github.com/lwCVer/DecoupleNet/releases/download/weights/DecoupleNet_D1.pth) |
| DecoupleNet D2 |     6.93     |   1.110   | [D2](https://github.com/lwCVer/DecoupleNet/releases/download/weights/DecoupleNet_D2.pth) |

   



## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models), [mmrotate](https://github.com/open-mmlab/mmrotate),  and [unetformer](https://github.com/WangLibo1995/GeoSeg) repositories.

If you have any questions about this work, you can contact me. Email: 2858191255@qq.com.

Your star is the power that keeps us updating github.

## Citation
If you find this repository helpful, please consider citing:
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
