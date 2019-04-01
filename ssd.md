# SSD

论文出处：https://arxiv.org/abs/1512.02325

参考代码：https://github.com/weiliu89/caffe/tree/ssd

一句话总结：SSD 实现了端到端的one-stage检测方法，类比于多尺度多分类的Faster RCNN RPN。

## 网络结构

![](./images/2-1-1.png)

* 在开始的层中使用分类的网络，如VGG16，这部分称之为 base network

* 在 base network 的基础上进行物体的检测

  * Mult-scale feature map for detection

    如图所示：

    * 分类网络的最后一层为 32 * 32 * 512
    * 第一层卷积激活后得到，19 * 19 * 1024
    * 10 * 10 * 512 
    * 5 * 5 * 256
    * 3 * 3 * 256

  * Convolutional predictors for detection

    对于上面所列的每一层，用一系列的卷积核进行预测。

    * 假设本层的 m * n * p
    * 用 3 * 3 的卷积进行预测。
    * 对于每个位置预测，得到类别得分，和对 default bounding boxes 的偏移量

  * Default boxes and aspect ratio

    在每个feature map上的每个位置，预测 K 个 box，对于每个 box 预测 C 个类别的得分， 以及default bounding box的偏移量。

    ![](./images/2-1-2.png)

## 损失函数

* $L(x,c,l,g)=\frac{1}{N}\big(L_{conf}(x,c)+{\alpha}L_{loc}(x,l,g)\big)$
* $N$ 匹配到的default box数
* $L_{loc}(x,l,g)=\Sigma_{i{\in}Pos}^{N}\Sigma_{m{\in}\{cx,cy,h,w\}}x_{ij}^{k}{smooth}_{L_1}(l_i^m-\hat{g}_j^m)$
  * $\hat{g}_j^{cx}=(g_j^{cx}-d_{i}^{cx})/d_i^w$
  * $\hat{g}_j^{cy}=(g_j^{cy}-d_{i}^{cy})/d_i^w$
  * $\hat{g}_j^w=log(\frac{g_j^w}{d_i^w})$
  * $\hat{h}_j^w=log(\frac{g_j^h}{d_i^h})$
  * $(g^{cx},g^{cy},g^2,g^h)$ 是ground truth box
  * $(d^{cx},d^{cy},d^2,=d^h)$是default box
  * $(l^{cx},l^{cy},l^2,l^h)$是预测关于default box的偏移量
* $L_{conf}=-\Sigma_{i{\in}Pos}^{N}x_{ij}^plog(\hat{c}_i^p)-\Sigma_{i{\in}Neg}log{\hat{c}_i^0}$
* $\hat{c}_i^p=\frac{exp(c_i^p)}{\Sigma_{p}exp(c_i^p)}$

## 参考代码

见 mx-Detectron ssd 的目录

## 总结与思考

与Faster相比：

* 锚框不仅要做真实分类(num_classes + 1)，还有对边框的回归。
* 不仅在分类网络的最后一层做预测，而且进行下采样做多尺度的预测。

SSD 在多尺度上进行检测，那么是否能将多尺度融合，进而减少检测次数加快时间呢？

* 锚框的生成，这种生成方式是否合理，平面密铺了解一下，从左上角开始迭代生成。
* 目标函数，对于图片上任意一个框，与我生成的锚框的距离小于一个值。基于此生成平面密铺的锚框。
