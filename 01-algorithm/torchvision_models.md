- [图像分类模型](#图像分类模型)
  - [经典卷积网络](#经典卷积网络)
    - [ResNet 系列（如 ResNet18、ResNet50）](#resnet-系列如-resnet18resnet50)
    - [VGG 系列（如 VGG16、VGG19）](#vgg-系列如-vgg16vgg19)
    - [AlexNet](#alexnet)
  - [轻量化模型](#轻量化模型)
    - [MobileNet 系列（如 MobileNetV2、MobileNetV3）](#mobilenet-系列如-mobilenetv2mobilenetv3)
    - [SqueezeNet](#squeezenet)
  - [高性能模型](#高性能模型)
    - [EfficientNet 系列（B0-B7）](#efficientnet-系列b0-b7)
    - [Vision Transformer (ViT)](#vision-transformer-vit)
- [目标检测模型](#目标检测模型)
  - [两阶段检测器](#两阶段检测器)
    - [Faster R-CNN（如 ResNet50-FPN 版本）](#faster-r-cnn如-resnet50-fpn-版本)
    - [Mask R-CNN](#mask-r-cnn)
  - [单阶段检测器](#单阶段检测器)
      - [SSD (Single Shot MultiBox Detector)](#ssd-single-shot-multibox-detector)
- [图像分割模型](#图像分割模型)
  - [语义分割](#语义分割)
    - [DeepLab 系列（如 DeepLabV3+）](#deeplab-系列如-deeplabv3)
    - [FCN (Fully Convolutional Network)](#fcn-fully-convolutional-network)
  - [实例分割](#实例分割)
    - [Keypoint R-CNN](#keypoint-r-cnn)
- [视频理解模型](#视频理解模型)
  - [R3D (ResNet3D)](#r3d-resnet3d)
  - [C3D](#c3d)
- [选型建议](#选型建议)


以下是 torchvision.models 模块中可用模型的分类及典型应用场景的总结，结合其架构特点与适用场景进行说明：

# 图像分类模型

## 经典卷积网络

### ResNet 系列（如 ResNet18、ResNet50）

- 特点：引入残差连接，解决深层网络梯度消失问题，参数量适中。
- 场景：通用图像分类（如 MNIST、CIFAR-10、ImageNet）。

### VGG 系列（如 VGG16、VGG19）

- 特点：堆叠 3×3 卷积层，结构简单但参数量大。
- 场景：教学实验、需稳定特征提取的任务（如艺术品分类）。

### AlexNet

- 特点：首个深度 CNN 突破，参数量中等。
- 场景：入门级分类任务或历史模型复现。


## 轻量化模型

### MobileNet 系列（如 MobileNetV2、MobileNetV3）

- 特点：采用深度可分离卷积，计算效率高。
- 场景：移动端/嵌入式设备部署（如实时交通标志识别）。

### SqueezeNet

- 特点：参数量极低（约 1.2M），模型体积小。
- 场景：资源受限环境（如无人机图像处理）。

## 高性能模型

### EfficientNet 系列（B0-B7）

- 特点：复合缩放平衡深度、宽度和分辨率。
- 场景：高精度分类需求（如医学影像诊断）。

### Vision Transformer (ViT)

- 特点：基于自注意力机制，全局特征捕捉能力强。
- 场景：大规模数据集分类（如卫星图像分析）




# 目标检测模型

## 两阶段检测器

### Faster R-CNN（如 ResNet50-FPN 版本）

- 特点：含区域提议网络（RPN），精度高但速度较慢。
- 场景：工业质检、医学细胞检测。

### Mask R-CNN

- 特点：扩展 Faster R-CNN，支持实例分割。
- 场景：自动驾驶中的行人/车辆检测与分割。

## 单阶段检测器

#### SSD (Single Shot MultiBox Detector)

- 特点：多尺度特征图预测，速度与精度平衡。
- 场景：实时监控视频中的多目标跟踪。

# 图像分割模型

## 语义分割

### DeepLab 系列（如 DeepLabV3+）

- 特点：空洞卷积扩大感受野，支持多尺度上下文融合。
- 场景：街景图像中的道路/建筑分割。

### FCN (Fully Convolutional Network)

- 特点：全卷积结构，支持任意尺寸输入。
- 场景：遥感图像中的植被覆盖分析。

## 实例分割

### Keypoint R-CNN

- 特点：基于 Mask R-CNN，预测关键点坐标。
- 场景：人体姿态估计、动物行为分析。


# 视频理解模型

## R3D (ResNet3D)

- 特点：3D 卷积处理时序特征。
- 场景：短视频动作识别（如健身动作分类）。

## C3D

- 特点：基于 VGG 的 3D 扩展，提取时空特征。
- 场景：监控视频中的异常行为检测

# 选型建议

| 场景         | 推荐模型                    | 优势                 |
| :----------- | :-------------------------- | :------------------- |
| 移动端部署   | MobileNetV3, ShuffleNet     | 低功耗，小模型体积   |
| 高精度分类   | EfficientNet-B4, ViT-Large  | 平衡计算资源与准确率 |
| 实时检测     | SSD, YOLOv5（需第三方实现） | 高帧率处理           |
| 医学图像分割 | DeepLabV3+, UNet（第三方）  | 精细边缘捕捉         |