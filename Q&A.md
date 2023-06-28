# About SNN-MAE

## SNN 的两种残差连接

- MS: 脉冲矩阵仅为 0 1
- SEW: 脉冲矩阵为整数

## MAE 需要解决的问题

### 问题一：MAE系列方法能否在特征图上mask

原始MAE使用重建75%masked的图像，这一点被证明了无法在SNN中使用

原因为：SNN 输出无法对应自然图像分布

基本想法：mask feature

<!-- > 是否还有其他方案？比如 MaskFeat 似乎在HOG特征这种传统图像特征上做文章 -->

- 原图
- data2vec: nlp cv outlier

CV 不希望 outlier 

MAE ： 图像那里 75% mask ==》 图像信息稀疏 ==》 outlier

MAE：需要 outlier 

> ANN ViT 特征图 和 MAE 训完之后 ViT 的特征图， outlier
> ANN ViT 特征图 和 data2vec 训完之后 ViT 的特征图， outlier

SNN outlier + MAE outlier +++++++ ==》 CV 

NLP outlier 

SNN + data2vec +

1024 4096 1024 资源消耗

MLP self Attention

GAU FLASH ==> SNN

SNN Tranformer ==》 +++

预训练 transformer Linear Transformer     SNN  80

### 问题二：MAE系列在SNN上的一些特性

已经有一个在SEW的脉冲上进行mask的方案，最终精度为81.9，但是它牺牲了SNN需要的发放率

基本想法：在MS的膜电位或者脉冲上进行mask

遇到的问题：

- 在膜电位上mask会导致输出（使用ANN作为decoder）
- 在脉冲上mask会导致网络陷入decoder输出全0的局部最优，是否有有效的正则方案

### 问题三：MAE对模型结构的要求

目前使用的SNN-Transformer结构的归纳偏置十分强，归因于前五层卷积层，这是否会影响到预训练

> 或许可以借鉴Spark对卷积网络的预训练策略

### 问题四：MAE系列方法是否有方案可以直接判断预训练后encoder的表征能力，不需要重新训练那种

> linear probing

### 提点

- bigger
- 
