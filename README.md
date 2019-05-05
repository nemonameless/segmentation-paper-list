# segmentation-paper-list
# Segmentation
Collection of online resources about segmentation.

pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision

Format: [简称 - 全名 - 刊名 - 时间](paper超链接)[[code](code超连接)]

## Datasets

1. Panoramic Segmentation
     - [Cityscapes](https://www.cityscapes-dataset.com/)
     - [ADE20k](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
     - [Mapillary Vistas](https://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html)
     - [COCO](http://cocodataset.org)

## Challenges

- [COCO](http://cocodataset.org/#home)
- [ECCV2018 COCO + Mapillary](http://cocodataset.org/workshop/coco-mapillary-eccv-2018.html)

## Evaluation Metric

1. Semantic segmentation
   - mIoU
   - PA
   - MPA
   - FWIoU
   - speed
   - net size

2. [Panoptic segmentation](https://arxiv.org/abs/1801.00868)
   - segmentation quality(SQ)
   - detection quality(DQ)
   - PQ
 
3. Backgrond/Foreground Segmentation
   -  mean absolute error(MAE)
   -  F-measure



## Demos && Appllications

## Opensource Projects

- [MMDetection](https://github.com/open-mmlab/mmdetection)

## Online Resources

- [Paper list - awesome-panoptic-segmentation](https://github.com/Angzz/awesome-panoptic-segmentation)
- [Paper list - awesome-semantic-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
- [Paper list - Semantic-Segmentation_DL](https://github.com/tangzhenyu/SemanticSegmentation_DL)

## Papers & Documents

### Classical Segmentation

1. Graph Cut
   - Min-Cut / Max-Flow
   - [Graph Cuts - Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images - ICCV - 2001](http://www.csd.uwo.ca/~yuri/Papers/iccv01.pdf)
   - [GrabCut - Interactive Foreground Extraction using Iterated Graph Cuts - SIGGRAPH - 2004](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)
   - [NCut - Normalized Cuts and Image Segmentation - PAMI -2000](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf)
2. Contour Model
   - SNAKE
   - Level Set
3. Superpixel
   - [SLIC - SLIC Superpixels - EPFL Technical Report - 2010](http://www.kev-smith.com/papers/SLIC_Superpixels.pdf)
   - [LSC - Superpixel Segmentation using Linear Spectral Clustering - CVPR - 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_Superpixel_Segmentation_Using_2015_CVPR_paper.pdf)

### Backgrond/Foreground Segmentation
   - [Saliency Optimization from Robust Detection - CVPR - 2014](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Zhu_Saliency_Optimization_from_2014_CVPR_paper.pdf)
   - [Deeply supervised salient object detection with short connections - CVPR - 2017] (http://openaccess.thecvf.com/content_cvpr_2017/papers/Hou_Deeply_Supervised_Salient_CVPR_2017_paper.pdf) [[code-pytorch](https://github.com/AceCoooool/DSS-pytorch)][[code-caffe](https://github.com/Andrew-Qibin/DSS)]

## Semantic/Stuff Segmentation
### Effective

- [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network - CVPR - 2017](https://arxiv.org/abs/1703.02719)
- [Learning a Discriminative Feature Network for Semantic Segmentation - CVPR - 2018](https://arxiv.org/abs/1804.09337)
- [Pyramid Attention Network for Semantic Segmentation - ARXIV - 2018](https://arxiv.org/abs/1805.10180v3)
- [Context Encoding for Semantic Segmentation - CVPR - 2018](https://arxiv.org/abs/1803.08904) [[code](https://github.com/zhanghang1989/PyTorch-Encoding)]
- [DenseASPP for Semantic Segmentation in Street Scenes - CVPR - 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf) [[code](https://github.com/DeepMotionAIResearch/DenseASPP)]
- [PSANet: Point-wise Spatial Attention Network for Scene Parsing - ECCV - 2018](https://hszhao.github.io/papers/eccv18_psanet.pdf) [[code](https://github.com/hszhao/PSANet)]
- [ExFuse: Enhancing Feature Fusion for Semantic Segmentation - ECCV - 2018](https://arxiv.org/abs/1804.03821)
- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation(DeepLabv3+) - ECCV - 2018](https://arxiv.org/abs/1804.03821) [[code](https://github.com/tensorflow/models/tree/master/research/deeplab)]
- [Dual Attention Network for Scene Segmentation - CVPR - 2019](https://arxiv.org/abs/1809.02983) [[code](https://github.com/junfu1115/DANet)]
- [CCNet: Criss-Cross Attention for Semantic Segmentation - ARXIV - 2018](https://arxiv.org/abs/1811.11721) [[code](https://github.com/speedinghzl/CCNet)]
- [OCNet: Object Context Network for Scene Parsing - ARXIV - 2018](https://arxiv.org/abs/1809.00916) [[code](https://github.com/PkuRainBow/OCNet.pytorch)]
- [Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation - CVPR - 2019](https://arxiv.org/abs/1903.02120)

### Efficient
- [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation  - ARXIV - 2016](https://arxiv.org/abs/1606.02147) [[code](https://github.com/TimoSaemann/ENet)]
- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images - ECCV - 2018](https://arxiv.org/abs/1704.08545) [[code](https://github.com/hszhao/ICNet)]
- [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation - ECCV - 2018](https://arxiv.org/abs/1707.03718) [[code](https://github.com/e-lab/LinkNet)]
- [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation - ECCV - 2018](https://arxiv.org/abs/1808.00897) [[code](https://github.com/ycszen/TorchSeg)]
- [ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation - ECCV - 2018](https://arxiv.org/abs/1803.06815) [[code](https://github.com/sacmehta/ESPNet)]
- [CGNet: A Light-weight Context Guided Network for Semantic Segmentation - ARXIV - 2018](https://arxiv.org/abs/1811.08201) [[code](https://github.com/wutianyiRosun/CGNet)]


## Instance Segmentation
- [Mask R-CNN - ICCV - 2017](https://arxiv.org/abs/1703.06870)
- [Learning to Segment Every Thing - ARXIV - 2017](https://arxiv.org/abs/1711.10370)
- [MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features - CVPR - 2018](https://arxiv.org/abs/1712.04837)
- [Path Aggregation Network for Instance Segmentation - CVPR - 2018](https://arxiv.org/abs/1803.01534) [[code](https://github.com/ShuLiu1993/PANet)]
- [Affinity Derivation and Graph Merge for Instance Segmentation - ECCV - 2018](https://arxiv.org/abs/1811.10870)
- [Hybrid Task Cascade for Instance Segmentation - CVPR - 2019](https://arxiv.org/abs/1901.07518v1)
- [Mask Scoring R-CNN - CVPR - 2019](https://arxiv.org/abs/1903.00241) [[code](https://github.com/zjhuang22/maskscoring_rcnn)]

### Amodal Segmentation

- [Semantic Amodal Segmentation - CVPR - 2017](https://arxiv.org/pdf/1509.01329.pdf)
- [Learning to See the Invisible: End-to-End Trainable Amodal Instance Segmentation - ARXIV - 2018](https://arxiv.org/abs/1804.08864)
- [Amodal Instance Segmentation - ECCV - 2016](https://arxiv.org/abs/1604.08202)

### Panoramic Segmentation

- [Panoptic Segmentation - ARXIV - 2018](https://arxiv.org/abs/1801.00868)
- [Weakly- and Semi-Supervised Panoptic Segmentation - ECCV -2018](http://www.robots.ox.ac.uk/~tvg/publications/2018/0095.pdf) [[code]( https://github.com/qizhuli/Weakly-Supervised-Panoptic-Segmentation)]
- [Panoptic Segmentation with a Joint Semantic and Instance Segmentation Network - ARXIV - 2018](https://arxiv.org/abs/1809.02110)
- [COCO18WINNER - Panoptic - Megvii](http://presentations.cocodataset.org/ECCV18/COCO18-Panoptic-Megvii.pdf)  
- [Interactive Full Image Segmentation - ARXIV - 2018](https://arxiv.org/abs/1812.01888)
- [Attention-guided Unified Network for Panoptic Segmentation - ARXIV - 2018](https://arxiv.org/abs/1812.03904)
- [Learning to Fuse Things and Stuff - ARXIV - 2018]( https://arxiv.org/abs/1812.01192)
- [Panoptic Feature Pyramid Networks - ARXIV - 2019](http://cn.arxiv.org/pdf/1901.02446v1)
- [UPSNet: A Unified Panoptic Segmentation Network - ARXIV - 2019](https://arxiv.org/abs/1901.03784)
- [Attention-guided Unified Network for Panoptic Segmentation - CVPR - 2019](https://arxiv.org/abs/1812.03904)

### Video Segmentation
- [One-Shot Video Object Segmentation - CVPR - 2017](https://arxiv.org/abs/1611.05198)
- [Video Object Segmentation Without Temporal Information - T-PAMI - 2017](https://arxiv.org/abs/1709.06031v2)
- [Online Adaptation of Convolutional Neural Networks for Video Object Segmentation - BMVC - 2017](https://arxiv.org/abs/1706.09364v2)
- [Efficient Video Object Segmentation via Network Modulation - CVPR - 2018](https://arxiv.org/abs/1802.01218v1)
- [Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning - CVPR - 2018](https://arxiv.org/abs/1804.03131v1)
- [PReMVOS: Proposal-generation, Refinement and Merging for Video Object Segmentation - ACCV - 2018](https://arxiv.org/abs/1807.09190v2)
- [VideoMatch: Matching based Video Object Segmentation - ECCV - 2018](https://arxiv.org/abs/1809.01123v1)
- [A Generative Appearance Model for End-to-end Video Object Segmentation - ARXIV - 2018](https://arxiv.org/abs/1811.11611v2)
- [Meta Learning Deep Visual Words for Fast Video Object Segmentation - ARXIV - 2018](https://arxiv.org/abs/1812.01397v1)
- [FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation - CVPR - 2019](https://arxiv.org/abs/1902.09513)

### 3D Segmentation

### Related Topics -- Image Matting
- [Deep Image Matting - CVPR - 2017](https://arxiv.org/pdf/1703.03872.pdf)
- [A Perceptually Motivated Online Benchmark for Image Matting - CVPR - 2009](https://publik.tuwien.ac.at/files/PubDat_180666.pdf)
- [Resources - Alpha Matting Evaluation Website](http://www.alphamatting.com/index.html)



