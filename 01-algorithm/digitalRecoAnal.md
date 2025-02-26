- [数据集](#数据集)
  - [名称与来源](#名称与来源)
  - [内容与结构](#内容与结构)
  - [特点与用途](#特点与用途)
  - [优势与局限](#优势与局限)
- [模块导入](#模块导入)
  - [import torch](#import-torch)
  - [import torch.nn as nn](#import-torchnn-as-nn)
  - [import torch.optim as optim](#import-torchoptim-as-optim)
  - [from torch.utils.data import DataLoader](#from-torchutilsdata-import-dataloader)
    - [torch.utils.data:](#torchutilsdata)
    - [DataLoader:](#dataloader)
  - [from torchvision import datasets, transforms](#from-torchvision-import-datasets-transforms)
    - [torchvision:](#torchvision)
    - [datasets:](#datasets)
    - [transforms:](#transforms)
  - [from sklearn.metrics import accuracy\_score](#from-sklearnmetrics-import-accuracy_score)
    - [关键作用解释：](#关键作用解释)
    - [准确率的局限性：](#准确率的局限性)
    - [多分类任务：](#多分类任务)
    - [其他常用评估指标：](#其他常用评估指标)
- [定义数据预处理](#定义数据预处理)
- [加载MNIST数据集](#加载mnist数据集)
- [创建数据加载器](#创建数据加载器)
  - [DataLoader 的核心作用](#dataloader-的核心作用)
  - [关键参数详解](#关键参数详解)
- [定义简单的CNN模型](#定义简单的cnn模型)
  - [网络整体结构](#网络整体结构)
  - [各层详解与参数计算](#各层详解与参数计算)
    - [卷积层 conv1](#卷积层-conv1)
    - [池化层 max\_pool2d（第一次）](#池化层-max_pool2d第一次)
    - [卷积层 conv2](#卷积层-conv2)
    - [池化层 max\_pool2d（第二次）](#池化层-max_pool2d第二次)
    - [展平操作](#展平操作)
    - [全连接层 fc1](#全连接层-fc1)
    - [全连接层 fc2](#全连接层-fc2)
  - [前向传播流程](#前向传播流程)
  - [关键设计分析](#关键设计分析)
    - [卷积核与通道数](#卷积核与通道数)
    - [池化层](#池化层)
    - [全连接层](#全连接层)
    - [输出层设计](#输出层设计)
  - [改进建议](#改进建议)
- [初始化模型、损失函数和优化器](#初始化模型损失函数和优化器)
- [训练模型](#训练模型)
- [测试模型](#测试模型)
- [计算准确率](#计算准确率)
- [实际运行](#实际运行)
  - [优化后代码](#优化后代码)
  - [运行结果](#运行结果)


# 数据集

要使用PyTorch编写一个识别数字8的代码，我们可以使用MNIST数据集。MNIST数据集包含手写数字的图像，每个图像的大小为28x28像素。我们将构建一个简单的卷积神经网络（CNN）来识别数字8。


## 名称与来源

MNIST 是 Modified National Institute of Standards and Technology 的缩写，源自NIST的手写数字数据集，经调整后更适用于机器学习。

## 内容与结构

1. 数据量：包含60,000张训练图像和10,000张测试图像。
2. 图像格式：28×28像素的灰度图，像素值0（黑）至255（白），数字居中显示。
3. 标签：对应0-9的阿拉伯数字，作为监督学习的标注。

## 特点与用途

1. 入门友好：数据量适中、预处理完善（尺寸统一、居中），适合快速验证算法。
2. 基准测试：常用于图像分类模型的性能评估，如卷积神经网络（CNN）的早期测试。
3. 教学工具：广泛用于机器学习教程，帮助理解数据预处理、模型训练与评估流程。

## 优势与局限

- 优势：简洁干净，减少噪声干扰，便于聚焦模型设计。
- 局限：过于简单（单一背景、灰度图），难以反映现实场景的复杂性，促使进阶数据集（如Fashion-MNIST、CIFAR-10）的采用。




**以下是一个完整的代码示例解析：**

# 模块导入

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
```

## import torch

import torch 是 Python 中的一条导入语句，作用是引入 PyTorch 库（一个开源的深度学习框架）。PyTorch 由 Facebook 的 AI 研究团队开发，广泛用于构建和训练神经网络、处理张量（Tensor）数据，以及执行科学计算任务。

核心功能：

1. 张量（Tensor）操作：PyTorch 的核心是 torch.Tensor，类似于 NumPy 的数组，但支持 GPU 加速和自动微分（Autograd）。
2. 自动微分（Autograd）：torch.autograd 模块可自动计算梯度，是神经网络训练的基础。
3. 神经网络构建：torch.nn 模块提供了预定义的层（如全连接层、卷积层）和损失函数。
4. 优化器：torch.optim 包含优化算法（如 SGD、Adam），用于更新模型参数。
5. GPU 加速：通过 CUDA 支持，可以用 GPU 大幅加速计算。




##  import torch.nn as nn
import torch.nn as nn 是 PyTorch 中导入神经网络模块的常见方式。具体解释如下：

torch.nn: 这是 PyTorch 的神经网络模块，包含了构建神经网络所需的各种类和函数，如层（Linear, Conv2d 等）、激活函数（ReLU, Sigmoid 等）和损失函数（MSELoss, CrossEntropyLoss 等）。

as nn: 这是将 torch.nn 模块简化为 nn，方便在代码中使用。例如，nn.Linear 比 torch.nn.Linear 更简洁。



## import torch.optim as optim

torch.optim 是什么？

PyTorch 是一个深度学习框架，torch.optim 是它的子模块，专门提供优化器（优化算法）。

优化器的作用是根据模型在训练时的梯度（gradient）自动调整模型的参数（例如权重和偏置），使模型的预测结果逐渐接近真实值。


## from torch.utils.data import DataLoader

from torch.utils.data import DataLoader 是 PyTorch 中用于导入数据加载工具 DataLoader 的语句。DataLoader 是一个非常重要的工具，用于高效地加载和处理数据集，特别是在训练深度学习模型时。

### torch.utils.data:

这是 PyTorch 提供的一个模块，专门用于处理数据加载和预处理。

它包含了许多与数据集相关的工具，比如 Dataset（定义数据集）和 DataLoader（加载数据集）。

### DataLoader:

DataLoader 是一个迭代器，用于从数据集中批量加载数据。

它支持多线程数据加载、数据打乱（shuffling）、批量处理（batching）等功能，能够显著提高数据加载的效率。

通常与 Dataset 结合使用，Dataset 定义数据集的内容，而 DataLoader 负责加载数据。

主要参数

- dataset: 要加载的数据集（必须是 torch.utils.data.Dataset 的子类）。
- batch_size: 每个批次的大小（默认是 1）。
- shuffle: 是否在每个 epoch 打乱数据（默认是 False）。
- num_workers: 使用多少个子进程来加载数据（默认是 0，表示在主进程中加载）。
- drop_last: 如果数据集大小不能被 batch_size 整除，是否丢弃最后一个不完整的批次（默认是 False）。

## from torchvision import datasets, transforms

from torchvision import datasets, transforms 是 PyTorch 中用于导入 torchvision 库中两个重要模块的语句。torchvision 是 PyTorch 的一个扩展库，专门用于处理图像数据和计算机视觉任务。

具体解释

### torchvision:

torchvision 是 PyTorch 的官方扩展库，提供了许多与图像处理相关的工具和预训练模型。

它包括数据集（datasets）、图像变换（transforms）、模型（models）和工具函数（utils）等模块。

### datasets:

datasets 模块提供了许多常用的公开数据集，例如 MNIST、CIFAR-10、ImageNet 等。

这些数据集可以直接下载并使用，非常适合快速实验和原型开发。

示例数据集：

- torchvision.datasets.MNIST
- torchvision.datasets.CIFAR10
- torchvision.datasets.ImageFolder（用于加载自定义图像数据集）

### transforms:

transforms 模块提供了许多图像预处理和数据增强的工具。

它可以将图像转换为张量、归一化、裁剪、旋转、翻转等。

常用的变换：

- transforms.ToTensor()：将 PIL 图像或 NumPy 数组转换为 PyTorch 张量。
- transforms.Normalize()：对图像进行归一化。
- transforms.RandomHorizontalFlip()：随机水平翻转图像。
- transforms.Compose()：将多个变换组合在一起。

## from sklearn.metrics import accuracy_score

是 Python 中的一条导入语句，用于从 scikit-learn 库的 metrics 模块中导入 accuracy_score 函数。accuracy_score 是一个常用的评估指标，用于计算分类模型的准确率（即模型预测正确的样本占总样本的比例）。


### 关键作用解释：

accuracy_score 的作用：

用于评估分类模型的性能。

计算公式：准确率=预测正确的样本数/总样本数
​
适用于二分类和多分类任务。

参数说明：

- y_true：真实标签（ground truth）。
- y_pred：模型预测的标签。
- normalize：是否返回比例（默认 True，返回准确率；False 时返回正确样本数）。
- sample_weight：样本权重（可选）。

示例代码：

```
from sklearn.metrics import accuracy_score

# 真实标签
y_true = [0, 1, 1, 0, 1, 0, 1, 0]
# 模型预测标签
y_pred = [0, 1, 0, 0, 1, 1, 1, 0]

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"准确率: {accuracy:.2f}")  # 输出: 准确率: 0.75
```

常见问题及解决方法：
报错 ModuleNotFoundError: No module named 'sklearn'：

原因：未安装 scikit-learn。

解决：通过以下命令安装：

```
pip install scikit-learn
```

### 准确率的局限性：

当类别不平衡时（例如 90% 的样本属于类别 A，10% 属于类别 B），准确率可能无法反映模型的真实性能。

此时可以使用其他指标，如 F1-score、ROC-AUC 等。

### 多分类任务：

accuracy_score 也适用于多分类任务，只需确保 y_true 和 y_pred 的标签一致。


### 其他常用评估指标：

| 指标名称  | 作用                                         | 导入方式                                    |
| :-------- | :------------------------------------------- | :------------------------------------------ |
| F1-score  | 平衡精确率和召回率（适合不平衡数据）         | from sklearn.metrics import f1_score        |
| Precision | 精确率（预测为正类的样本中实际为正类的比例） | from sklearn.metrics import precision_score |
| Recall    | 召回率（实际为正类的样本中被正确预测的比例） | from sklearn.metrics import recall_score    |
| ROC-AUC   | ROC 曲线下面积（适合二分类）                 | from sklearn.metrics import roc_auc_score   |

示例：多分类任务的准确率计算

```
from sklearn.metrics import accuracy_score

# 多分类任务
y_true = [0, 1, 2, 2, 1, 0]
y_pred = [0, 2, 2, 1, 1, 0]

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"准确率: {accuracy:.2f}")  # 输出: 准确率: 0.67
```

总结：

accuracy_score 是评估分类模型性能的简单而直观的指标。

如果数据不平衡，建议结合其他指标（如 F1-score、ROC-AUC）综合评估模型。

确保 y_true 和 y_pred 的格式一致（例如都是列表或 NumPy 数组）。


# 定义数据预处理

```
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

**transforms.Compose**

- 作用：将多个图像变换操作按顺序组合成一个整体，依次执行。
- 流程：先执行ToTensor()，再执行Normalize()。

**transforms.ToTensor()**


功能：转张量 + 归一化到[0,1]

- 将输入的图像（如PIL图像或NumPy数组）转换为PyTorch张量（torch.Tensor）。
- 自动将像素值范围从[0, 255]缩放到[0.0, 1.0]。
- 调整张量维度顺序为(C, H, W)（通道数 × 高度 × 宽度）。例如，RGB图像的形状会从(H, W, 3)变为(3, H, W)。

输入：支持PIL图像（PIL.Image）或NumPy数组。

输出：形状为(C, H, W)的张量，值域为[0.0, 1.0]。


**transforms.Normalize((0.5,), (0.5,))**


功能：对张量进行标准化到[-1,1]

公式为：

$$
output=\frac{input-mean}{std}
$$

其中：

mean=(0.5,)：每个通道的均值。

std=(0.5,)：每个通道的标准差。

具体计算：

输入张量的值域为[0.0, 1.0]（来自ToTensor()的输出）。

标准化后的值域变为：

$$
\frac{[0.0,1.0]-0.5}{0.5}=[-1.0,1.0]
$$

参数说明：


如果输入是单通道（如灰度图），参数为单值元组(0.5,)。

如果是多通道（如RGB），需为每个通道指定均值和标准差，例如：

```
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 三通道
```









# 加载MNIST数据集


```
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

**train_dataset = datasets.MNIST(...)**

功能：加载MNIST训练集。

参数解析：

root='./data'

- 数据集存储的根目录路径。如果目录中不存在数据集，会自动下载到此路径。
- 示例：数据会保存在./data/MNIST/raw/和./data/MNIST/processed/中。

train=True

- 指定加载训练集（共60,000张图像）。
- 若设置为train=False，则加载测试集。

download=True

- 如果root路径下没有数据集，则从网络下载。
- 若已存在数据集，跳过下载。

transform=transform

- 应用之前定义的预处理流水线（transform变量，包含ToTensor和Normalize操作）。
- 输入图像会被转换为张量并标准化到[-1, 1]范围。

**. test_dataset = datasets.MNIST(...)**

功能：加载MNIST测试集。

参数差异：

train=False

- 指定加载测试集（共10,000张图像）。
- 其他参数与训练集一致，确保测试集与训练集使用相同的预处理逻辑。




# 创建数据加载器

```
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

## DataLoader 的核心作用

- 功能：将数据集包装为可迭代对象，支持批量加载、数据打乱、多进程加速等。
- 输入：Dataset 对象（如 train_dataset 和 test_dataset）。
- 输出：按批次生成 (数据, 标签) 元组，可直接用于模型训练或测试。

## 关键参数详解

dataset=train_dataset

- 指定加载的训练数据集（已通过 transforms 预处理）。
- 数据格式：每个样本为 (图像张量, 标签)，其中图像形状为 (1, 28, 28)（单通道28×28像素）。

batch_size=64

- 每批次加载64张图像及其标签。
- 影响：
  - 内存占用：较大的 batch_size 占用更多内存，但可能加速训练。
  - 梯度稳定性：较大的批次使梯度估计更准确，但可能降低模型泛化能力。

- 常见选择：根据硬件资源调整（如32、64、128）。

shuffle=True

- 在每个训练周期（epoch）开始时，打乱数据顺序。
- 目的：防止模型因数据顺序产生偏差（如先学习某一类别的样本）。
- 仅用于训练集：测试集无需打乱。











# 定义简单的CNN模型

```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)  # 输出为1，表示是否为数字8

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```


class CNN(nn.Module)：定义名为 CNN 的类，继承 PyTorch 的神经网络基类 nn.Module。

def \__init__(self)：初始化函数，定义网络结构。

super(CNN, self).\__init__()：调用父类 nn.Module 的初始化方法。

self.conv1：第一个卷积层，输入通道数 1（灰度图），输出通道数 32，卷积核大小 3×3，步长 1，填充 1（保持输入输出尺寸一致）。

self.conv2：第二个卷积层，输入通道数 32，输出通道数 64，参数与 conv1 相同。

self.fc1：第一个全连接层，输入维度 64*7*7（两次池化后的特征图展平），输出维度 128。

self.fc2：输出层，输入维度 128，输出维度 1（二分类任务，如判断是否为数字“8”）。






## 网络整体结构


该CNN用于MNIST手写数字的二分类任务（判断是否为数字8），包含两个卷积层和两个全连接层，结构如下：

```
输入层 (Input)
↓ 尺寸: [Batch, 1, 28, 28]
|
├─ **Conv1层**  
│  ├─ 类型: Conv2d  
│  ├─ 参数: in=1, out=32, kernel=3×3, stride=1, padding=1  
│  ├─ 输出尺寸: [Batch, 32, 28, 28]  
│  └─ 激活函数: ReLU  
↓
|
├─ **MaxPool1层**  
│  ├─ 类型: MaxPool2d  
│  ├─ 参数: kernel=2×2, stride=2  
│  └─ 输出尺寸: [Batch, 32, 14, 14]  
↓
|
├─ **Conv2层**  
│  ├─ 类型: Conv2d  
│  ├─ 参数: in=32, out=64, kernel=3×3, stride=1, padding=1  
│  ├─ 输出尺寸: [Batch, 64, 14, 14]  
│  └─ 激活函数: ReLU  
↓
|
├─ **MaxPool2层**  
│  ├─ 类型: MaxPool2d  
│  ├─ 参数: kernel=2×2, stride=2  
│  └─ 输出尺寸: [Batch, 64, 7, 7]  
↓
|
├─ **展平层 (Flatten)**  
│  └─ 输出尺寸: [Batch, 64×7×7] → [Batch, 3136]  
↓
|
├─ **FC1层 (全连接层)**  
│  ├─ 类型: Linear  
│  ├─ 参数: in=3136, out=128  
│  └─ 激活函数: ReLU  
↓
|
└─ **FC2层 (输出层)**  
   ├─ 类型: Linear  
   ├─ 参数: in=128, out=1  
   └─ 输出值: Logits (未激活的二分类结果)
```


## 各层详解与参数计算

### 卷积层 conv1

```
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
```


输入：单通道灰度图，尺寸为 (1, 28, 28)（MNIST图像）。

输出：32通道特征图，尺寸保持 28×28。

计算逻辑：

$$
outputSize=\frac{inputSize+2×padding-kernelSize}{stride}+1=\frac{28+2×1-3}{1}+1=28
$$


参数数量：$(3×3×1)×32+32=896$


### 池化层 max_pool2d（第一次）

```
x = torch.max_pool2d(x, 2)  # kernel_size=2, stride=2（默认）
```

输入：32通道的 28×28 特征图。

输出：32通道的 14×14 特征图，输出尺寸：28/2=14。







### 卷积层 conv2

```
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
```

输入：32通道的 14×14 特征图。

输出：64通道的 14×14 特征图（计算方式同 conv1）。

参数数量：$(3×3×32)×64+64=18496$


### 池化层 max_pool2d（第二次）

```
x = torch.max_pool2d(x, 2)
```

输入：64通道的 14×14 特征图。

输出：64通道的 7×7 特征图。


### 展平操作

```
x = x.view(x.size(0), -1)  # 形状变为 (batch_size, 64*7*7=3136)
```

作用：将三维特征图（64×7×7）转换为一维向量，供全连接层处理。


### 全连接层 fc1

```
self.fc1 = nn.Linear(64 * 7 * 7, 128)
```

输入：3136维向量（每个样本）。

输出：128维向量。

参数数量：$3136×128+128=401536$


### 全连接层 fc2

```
self.fc2 = nn.Linear(128, 1)  # 输出为1，表示是否为数字8
```

## 前向传播流程

```
    def forward(self, x):
        x = torch.relu(self.conv1(x))       # 卷积1 + ReLU激活
        x = torch.max_pool2d(x, 2)          # 最大池化1（下采样）
        x = torch.relu(self.conv2(x))       # 卷积2 + ReLU激活
        x = torch.max_pool2d(x, 2)          # 最大池化2（下采样）
        x = x.view(x.size(0), -1)           # 展平多维特征图为一维向量
        x = torch.relu(self.fc1(x))        # 全连接1 + ReLU激活
        x = self.fc2(x)                     # 输出层（无激活函数）
        return x
```


x = torch.relu(self.conv1(x))：
对输入 x 进行第一次卷积操作，然后通过 ReLU 激活函数。

输入尺寸：假设输入为 (batch_size, 1, 28, 28)（MNIST图像），输出尺寸为 (batch_size, 32, 28, 28)。

x = torch.max_pool2d(x, 2)：
使用 2×2 的窗口进行最大池化，步长默认与窗口大小相同（即 2）。

输出尺寸：(batch_size, 32, 14, 14)（长宽各减半）。

x = torch.relu(self.conv2(x))：
第二次卷积操作，输出通道数 64，尺寸保持 14×14。

输出尺寸：(batch_size, 64, 14, 14)。

x = torch.max_pool2d(x, 2)：
第二次最大池化，输出尺寸为 (batch_size, 64, 7, 7)。

x = x.view(x.size(0), -1)：
将多维特征图展平为一维向量，x.size(0) 保留 batch 维度，-1 自动计算展平后的长度。

展平后尺寸：(batch_size, 64*7*7) = (batch_size, 3136)。

x = torch.relu(self.fc1(x))：

全连接层 fc1 后接 ReLU 激活，输出维度 128。

x = self.fc2(x)：
输出层直接返回 logits（未激活的数值），适用于二分类任务（如使用 BCEWithLogitsLoss 损失函数）。













## 关键设计分析

### 卷积核与通道数

通道增长：1 → 32 → 64，逐步提取复杂特征。

小卷积核：3×3 卷积核平衡计算效率和特征捕获能力。

Padding策略：padding=1 保持特征图尺寸不变，避免信息丢失。

### 池化层

最大池化：使用 2×2 窗口，减少空间维度（28→14→7），增强平移不变性。

步长：默认 stride=2，与池化窗口大小一致。

### 全连接层

维度压缩：3136 → 128 → 1，逐步压缩信息至最终分类结果。

激活函数：仅在全连接层前使用ReLU，避免梯度消失。

### 输出层设计

无激活函数：直接输出logits（原始分数），配合 BCEWithLogitsLoss 使用（内部集成Sigmoid与交叉熵）。

二分类适配：单神经元输出，简化模型结构。


## 改进建议

添加 Dropout：在全连接层后加入 nn.Dropout() 防止过拟合。

```
self.dropout = nn.Dropout(0.5)  # 添加到 __init__
x = self.dropout(torch.relu(self.fc1(x)))  # 修改 forward
```

使用 BatchNorm：加速训练并提升稳定性。

```
self.bn1 = nn.BatchNorm2d(32)  # 添加到 __init__
x = torch.relu(self.bn1(self.conv1(x)))  # 修改 forward
```

全局平均池化替代全连接层：减少参数量（现代网络常用）。

```
self.gap = nn.AdaptiveAvgPool2d((1,1))  # 添加到 __init__
x = self.gap(x)  # 替换展平操作
x = x.view(x.size(0), -1)
```






# 初始化模型、损失函数和优化器



```
model = CNN()
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


解析：

CNN() 调用之前定义的 CNN 类的构造函数，初始化网络结构（卷积层、池化层、全连接层等）。

默认设备：模型参数和数据会存储在CPU上；若需使用GPU训练，需显式指定（如 model.to('cuda')）。

关键点：

模型初始化后，可通过 print(model) 查看网络结构。

模型参数（权重和偏置）会随机初始化，训练过程中通过优化器逐步更新。


解析：

BCEWithLogitsLoss：结合 Sigmoid 激活和二分类交叉熵损失（Binary Cross Entropy, BCE）的复合函数。

数学公式：

$$
Loss=-\frac{1}{N}\sum_{i=1}^{n}{[{{y}_{i}}⋅log(σ({{z}_{i}}))+(1+{{y}_{i}})⋅log(1-σ({{z}_{i}}))]}
$$
​
其中 ${z}_{i}$ 是模型输出的 logits（未经过 Sigmoid），σ 是 Sigmoid 函数。

数值稳定性：内部优化计算，避免 Sigmoid 的极端值导致数值溢出。

适用场景：二分类任务（如判断是否为数字“8”），输出层直接返回 logits（如代码中 fc2 层无激活函数）。








# 训练模型

```
num_epochs = 5
```
作用：定义训练的总轮数（epoch），即完整遍历训练集的次数。

解析：

此处设置为 5，表示模型将在训练数据上迭代训练 5 次。


```
for epoch in range(num_epochs):
    model.train()
```


常见调整：根据任务复杂度调整轮数（简单任务可能只需 10 轮，复杂任务可能需数百轮）。

for epoch in range(num_epochs)：遍历每一轮训练（从 0 到 num_epochs-1）。

model.train()：将模型设置为训练模式，启用 Dropout、BatchNorm 等层的训练行为。


```
    for images, labels in train_loader:
        # 将标签转换为二进制：1表示数字8，0表示其他数字
        labels = (labels == 8).float().view(-1, 1)
```

for images, labels in train_loader：
从数据加载器 train_loader 中按批次（batch）加载数据，每批次包含图像 images 和标签 labels。

输入假设：假设 train_loader 输出标签为整数（如 MNIST 的标签 0~9）。

labels = (labels == 8).float().view(-1, 1)：
将标签转换为二分类任务的二进制形式：

labels == 8：生成布尔张量（True/False），对应是否为数字 8。

.float()：将布尔值转换为浮点数（1.0 表示是数字 8，0.0 表示其他）。

.view(-1, 1)：调整形状为 [batch_size, 1]，与模型输出的 logits 维度匹配。



```
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
```

outputs = model(images)：将图像输入模型，得到预测输出 outputs（logits）。

loss = criterion(outputs, labels)：计算损失值，criterion 为 BCEWithLogitsLoss，直接处理 logits 和二进制标签。


```
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```    

optimizer.zero_grad()：清空优化器中所有参数的梯度，防止梯度累积。

loss.backward()：反向传播计算梯度（自动微分）。

optimizer.step()：根据梯度更新模型参数（执行优化步骤）。

```
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

loss.item()：从张量中提取标量损失值（转换为 Python 浮点数）。

.4f：格式化输出，保留 4 位小数。

注意：此处打印的是当前 epoch 最后一个 batch 的损失，而非整个 epoch 的平均损失。
改进建议：在 epoch 内累积损失并计算平均值，更准确反映训练状态。







# 测试模型

```
model.eval()
```

作用：切换模型至评估模式，关闭训练相关的特定层行为（如 Dropout、BatchNorm 的随机性）。

关键点：

必需操作：确保评估阶段结果稳定，避免因随机性导致预测波动。

对比：训练时需调用 model.train() 恢复随机行为。


```
all_preds = []
all_labels = []
```

作用：创建空列表，用于存储所有批次的预测结果和真实标签。

解析：

all_preds：保存模型输出的二分类预测结果（0 或 1）。

all_labels：保存二值化后的真实标签（0 或 1）。


```
with torch.no_grad():
```

作用：在上下文管理器内禁用自动梯度计算，减少内存占用并加速推理。

原理：

torch.no_grad()：关闭 requires_grad 标志，避免前向传播时构建计算图。

节省资源：适用于无需反向传播的评估或推理阶段。


```
    for images, labels in test_loader:
        # 将标签转换为二进制
        labels = (labels == 8).float().view(-1, 1)
```

逐行解析：

for images, labels in test_loader：从测试数据加载器中按批次加载图像和原始标签。

labels = (labels == 8).float().view(-1, 1)：

二值化：将标签转换为 1.0（是数字8）或 0.0（其他数字）。

维度调整：view(-1, 1) 确保标签形状为 [batch_size, 1]，与模型输出对齐。


```
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5  # 将输出转换为二进制预测
```

outputs = model(images)：模型对输入图像进行前向传播，输出 logits（未激活的原始值）。

preds = torch.sigmoid(outputs) > 0.5：

概率转换：通过 Sigmoid 将 logits 映射为概率值（范围 [0, 1]）。

阈值判断：概率 > 0.5 视为正类（是数字8），否则为负类。


```
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
```

.cpu()：将张量从 GPU 移至 CPU（若模型在 GPU 上运行）。

.numpy()：将张量转换为 NumPy 数组。

extend()：将当前批次的预测和标签添加到全局列表中。





# 计算准确率

```
accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
```


第一行：accuracy = accuracy_score(all_labels, all_preds)

​函数作用​：调用 accuracy_score 函数计算模型预测的准确率。这是分类任务中常用的评估指标，表示预测正确的样本占总样本的比例。

​参数解析​：

- all_labels：真实标签列表或数组，通常从测试数据集的标签中获取。
- all_preds：模型预测的标签列表或数组，需与真实标签一一对应。

​返回值​ accuracy：浮点数（float），取值范围为 [0, 1]，例如 0.95 表示95%的准确率。

​注意事项​

- 需确保 all_labels 和 all_preds 的维度一致，否则会报错。
- 此函数来自 sklearn.metrics 库，需提前导入：from sklearn.metrics import accuracy_score。

​第二行：print(f'Test Accuracy: {accuracy * 100:.2f}%')

​格式化输出​：使用 f-string（格式化字符串）将准确率转换为百分比形式。

{accuracy * 100:.2f}：将 accuracy 乘以 100 转换为百分比，并通过 :.2f 保留两位小数（例如 95.23%）。
​功能分解​

- accuracy * 100：将小数形式的准确率转换为百分比（如 0.9523 → 95.23）。
- :.2f：格式化指令，表示保留两位小数。
- %：在字符串末尾添加百分号。

​输出示例​

若 accuracy = 0.9523，输出结果为：Test Accuracy: 95.23%。

​代码整体逻辑

- ​计算准确率：通过对比真实标签和预测标签，量化模型性能。
- ​结果展示：将数值转换为更易读的百分比形式输出。

# 实际运行

## 优化后代码

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score



# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True, 
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=64, 
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=1000, 
    shuffle=False)



# 定义简单的CNN模型
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 1)  # 输出为1，表示是否为数字8
        
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)  # 展平
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)        
#         return x



# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64, 128)  # 修正输入维度为 64
#         self.fc2 = nn.Linear(128, 1)
#         self.dropout = nn.Dropout(0.5)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.max_pool2d(x, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2)
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(torch.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x
    
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, 128)  # 输入维度修正为64
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)  # 新增BN层
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.bn2(self.conv2(x)))  # 添加BN和ReLU
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)  # 添加Sigmoid
    
    
    

    
    
# 初始化模型、损失函数和优化器
model = CNN()

criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        # 将标签转换为二进制：1表示数字8，0表示其他数字
        labels = (labels == 8).float().view(-1, 1)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        # 将标签转换为二进制
        labels = (labels == 8).float().view(-1, 1)
        
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5  # 将输出转换为二进制预测
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算准确率
accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


# 保存模型
torch.save(model.state_dict(), "mnist_relu_model.pth")
```

## 运行结果

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital8recog.jpg"></div>


显然增加训练次数并不会显著提高识别成功率



