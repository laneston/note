- [神经网络模型示例](#神经网络模型示例)
  - [nn.Module](#nnmodule)
    - [nn.Module 的核心意义](#nnmodule-的核心意义)
    - [关键功能详解](#关键功能详解)
  - [super()](#super)
- [网络结构详解](#网络结构详解)
  - [卷积层1 (conv1)](#卷积层1-conv1)
    - [参数解析](#参数解析)
    - [输出尺寸计算](#输出尺寸计算)
    - [输出张量的维度](#输出张量的维度)
  - [卷积层2 (conv2)](#卷积层2-conv2)
  - [最大池化](#最大池化)
    - [核心作用](#核心作用)
    - [参数解析](#参数解析-1)
    - [与其他池化方法的对比](#与其他池化方法的对比)
  - [Dropout正则化](#dropout正则化)
    - [核心作用](#核心作用-1)
    - [参数解析](#参数解析-2)
    - [输入输出示例](#输入输出示例)
    - [设计意义](#设计意义)
  - [全连接层 (fc1和fc2)](#全连接层-fc1和fc2)
    - [参数解析](#参数解析-3)
    - [数学操作](#数学操作)
    - [功能作用](#功能作用)
    - [设计意义](#设计意义-1)
  - [激活与输出](#激活与输出)
- [案例结果](#案例结果)
  - [输入](#输入)
  - [训练5轮模型识别效果](#训练5轮模型识别效果)
  - [训练15轮模型识别效果](#训练15轮模型识别效果)
  - [结论](#结论)



# 神经网络模型示例

```
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 卷积层1：输入通道1（灰度图），输出通道32，3x3卷积核，步长1
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 卷积层2：输入通道32，输出通道64，3x3卷积核，步长1
        self.dropout1 = nn.Dropout2d(0.25)   # Dropout层：防止过拟合，丢弃率25%
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)      # 全连接层1：输入9216维，输出128维（需根据特征图尺寸计算）
        self.fc2 = nn.Linear(128, 10)        # 全连接层2：输出10维（对应0-9数字分类）

    def forward(self, x):
        x = self.conv1(x)                                # 第一次卷积 + ReLU激活
        x = nn.functional.relu(x)
        x = self.conv2(x)                                # 第二次卷积 + ReLU激活
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)               # 最大池化（下采样）
        x = self.dropout1(x)                             # Dropout正则化
        x = torch.flatten(x, 1)                          # 展平特征图
        x = self.fc1(x)                                  # 全连接层1 + ReLU激活
        x = nn.functional.relu(x)
        x = self.dropout2(x)                             # 第二次Dropout（丢弃率50%）
        x = self.fc2(x)                                  # 输出层 + log_softmax（配合NLLLoss损失函数）
        return nn.functional.log_softmax(x, dim=1)
```



这个DigitRecognizer类是一个用于手写数字识别的卷积神经网络（CNN）模型，专为处理类似MNIST这样的28x28像素灰度图像设计。以下是其核心作用及结构分析：

## nn.Module

在PyTorch中，nn.Module 是所有神经网络模块的基类，其核心意义在于为模型开发提供统一的结构化框架。以下是它的核心作用和设计意义的详细解析：

### nn.Module 的核心意义

| 功能           | 说明                                                           |
| :------------- | :------------------------------------------------------------- |
| 参数自动管理   | 自动追踪所有子模块的参数（nn.Parameter），无需手动维护参数列表 |
| 模块化网络构建 | 支持嵌套子模块（如层、块），形成层次化结构                     |
| 设备无缝迁移   | 通过.to(device)一键将模型参数和缓冲区移到CPU/GPU               |
| 模型序列化     | 提供state_dict()和load_state_dict()方法，支持模型保存与加载    |
| 计算图集成     | 与PyTorch的自动微分机制（Autograd）深度集成，自动处理反向传播  |
| 钩子函数支持   | 允许注册前向/反向传播的钩子，用于调试或自定义操作              |

### 关键功能详解

**(1)参数自动管理**

自动追踪机制：所有通过self.xxx = nn.Linear(...)定义的层参数会被自动注册

访问方式：

```
# 获取所有可训练参数
params = list(model.parameters())

# 获取特定子模块参数
conv_params = list(model.conv1.parameters())
```

**(2)模块化构建**

```
class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
    
    def forward(self, x):
        return self.bn(self.conv(x))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = MyBlock()  # 嵌套子模块
        self.pool = nn.MaxPool2d(2)
```

**(3)设备迁移**

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)  # 所有参数自动移至指定设备
```

**(4)模型序列化**

```
# 保存模型
torch.save(model.state_dict(), "model.pth")

# 加载模型
model.load_state_dict(torch.load("model.pth"))
```

## super()

Python 2：需显式传递子类名和实例，即 super(子类名, self)。

Python 3：可简化为 super().__init__()，解释器会自动推断。
示例中的写法 super(DigitRecognizer, self).__init__() 是为了兼容旧版本，但现代PyTorch代码通常直接写 super().__init__()。





# 网络结构详解


## 卷积层1 (conv1)

在PyTorch中，self.conv1 = nn.Conv2d(1, 32, 3, 1) 定义了一个二维卷积层，以下是各参数的详细解析及其作用：


- 输入：1通道（灰度图）
- 输出：32通道，使用3x3卷积核，步长1
- 作用：检测初级特征（如边缘、线条）。


### 参数解析

**in_channels=1**

- 含义：输入数据的通道数。
- 示例值：1（适用于灰度图像，如MNIST数据集）。
- 作用：指定输入特征图的通道数。对于RGB彩色图像，应设为3。

**out_channels=32**

- 含义：输出数据的通道数，即卷积核的数量。
- 作用：每个卷积核生成一个特征图，32 表示生成32个不同的特征图，用于捕捉输入的不同特征。

**kernel_size=3**

- 含义：卷积核的大小。
- 示例值：3（表示3x3的卷积核）。
- 作用：决定感受野的大小。较小的卷积核（如3x3）能捕捉局部特征，参数更少，计算效率高。

**stride=1**

- 含义：卷积核滑动的步长。
- 作用：控制输出特征图的尺寸。步长越大，输出尺寸越小。步长为1时，输出尺寸最大。


### 输出尺寸计算

假设输入为单通道的28x28图像（如MNIST），无填充（默认padding=0），则输出尺寸计算如下：

$$
outputSize=\frac{inputSize-kernelSize+2×padding}{stride}+1
$$

代入参数：$(28-3+2×0)/1+1=26$

**输出形状：** `(batch_size, 32, 26, 26)`,32个通道（特征图），每个特征图大小为26x26。

### 输出张量的维度

输出张量的形状为：`(batch_size, out_channels, H_out, W_out)`


- batch_size：输入数据的样本数量（即一个批次中的图像数量）
- out_channels：卷积层定义的输出通道数（即卷积核的数量）
- H_out：输出特征图的高度
- W_out：输出特征图的宽度

若需保持尺寸不变，需设置 padding=(kernel_size-1)//2。



## 卷积层2 (conv2)

- 输入：32通道
- 输出：64通道，3x3卷积核，步长1
- 作用：捕获更复杂的特征（如曲线、组合形状）。

**输出形状：** `(batch_size, 64, 24, 24)`,46个通道（特征图），每个特征图大小为24x24。


## 最大池化

窗口大小2x2，将特征图尺寸减半（如24x24 → 12x12），保留显著特征并减少计算量。

在PyTorch中，nn.functional.max_pool2d(x, 2) 是一个用于二维最大池化（Max Pooling）操作的函数，其核心意义和功能如下：


### 核心作用

- 降维压缩：通过保留局部区域的最大值，逐步减小特征图的空间尺寸（高度和宽度），降低计算复杂度。
- 特征不变性增强：对平移、旋转、缩放等轻微形变保持鲁棒性，保留最显著的特征。
- 防止过拟合：减少参数量的同时抑制噪声响应。

### 参数解析

- x：输入张量，形状为 (batch_size, channels, height, width)
- 2：池化窗口大小（等价于 kernel_size=2），同时隐含：
  - stride=2（步长与窗口大小一致）
  - 默认无填充（padding=0）

```
nn.functional.max_pool2d(
    input = x,
    kernel_size = 2,  # 池化窗口尺寸
    stride = 2,       # 滑动步长
    padding = 0,      # 边缘填充
    dilation = 1,     # 空洞卷积（默认为1，即无空洞）
)
```

```
[[1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 10, 11, 12],
 [13, 14, 15, 16]]
```

```
[[6, 8],
 [14, 16]]
```

原理：每个 2×2 窗口取最大值（如左上窗口 [1,2,5,6] → 6）

###  与其他池化方法的对比


| 池化类型         | 计算方式       | 适用场景                       |
| :--------------- | :------------- | :----------------------------- |
| Max Pooling      | 取窗口内最大值 | 纹理、边缘等显著特征提取       |
| Average Pooling  | 取窗口内平均值 | 平滑特征（如图像分类最后一层） |
| Adaptive Pooling | 自适应输出尺寸 | 输入尺寸不固定的模型           |



## Dropout正则化

- dropout1（25%丢弃率）：在池化后应用，防止过拟合。
- dropout2（50%丢弃率）：在全连接层前应用，增强泛化能力。



在PyTorch中，self.dropout1 = nn.Dropout2d(0.25) 是一个用于正则化的层，其核心意义是通过随机丢弃特征图的整个通道来防止过拟合。以下是详细解析：

### 核心作用


| 功能           | 说明                                                               |
| :------------- | :----------------------------------------------------------------- |
| 通道级随机失活 | 以通道为单位随机丢弃整个特征图（例如，丢弃某几个通道的所有激活值） |
| 防止过拟合     | 迫使网络不过度依赖单个通道的特征，增强模型泛化能力                 |
| 特征独立性增强 | 减少通道之间的共适应性（co-adaptation）                            |

### 参数解析


0.25（p=0.25）：表示每个通道在训练时有 25% 的概率被完全置零，剩余通道的值会被按比例放大（保持期望值不变）。

数学公式：

$$
output=\left \{{\begin{matrix}0&概率为p\\\frac{input}{1-p}&概率为1-p\end{matrix}}\right .
$$

### 输入输出示例

假设输入特征图形状为 (batch_size=2, channels=4, height=3, width=3)：

```
# 输入特征图（4个通道）
Channel 0: [[1,2,3], [4,5,6], [7,8,9]]  
Channel 1: [[10,11,12], [13,14,15], [16,17,18]]  
Channel 2: [[19,20,21], [22,23,24], [25,26,27]]  
Channel 3: [[28,29,30], [31,32,33], [34,35,36]]  
```

应用 Dropout2d(p=0.25) 后可能的输出（假设丢弃第1和第3通道）：

```
Channel 0: [[1/(1-0.25), 2/(0.75), ...], ...]  → 放大1.33倍  
Channel 1: [[0, 0, 0], ...]                     → 全部置零  
Channel 2: [[19/(0.75), 20/(0.75), ...], ...]    → 放大1.33倍  
Channel 3: [[0, 0, 0], ...]                     → 全部置零  
```

### 设计意义

- 适配卷积特性：卷积层的输出通道通常代表不同特征，丢弃整个通道更符合卷积特征的结构化特性。
- 保持空间一致性：避免随机丢弃单个像素破坏特征图的空间关联性。
- 计算高效性：通道级操作比元素级操作更快。


放置位置：通常在卷积层后、激活函数前使用

参数选择：

- 浅层网络：较小值（如0.2）
- 深层网络：较大值（如0.5）
- 数据量少时：增大丢弃率


## 全连接层 (fc1和fc2)

nn.Linear 的原理是通过线性变换（矩阵乘法 + 偏置）实现特征空间的映射

在PyTorch中，self.fc1 = nn.Linear(9216, 128) 是一个全连接层（线性层），其核心意义和功能如下：

- fc1：将展平后的9216维特征（64通道×12×12）映射到128维，进一步整合特征。
- fc2：输出10维，对应0-9分类结果。



### 参数解析

**输入维度（in_features=9216）：** 表示展平后输入数据的特征总数。该值通常由前一层的输出形状决定。例如，若前一层的输出为 (batch_size, 64, 12, 12)，则展平后的特征数为 $64×12×12=9216$ 。

**输出维度（out_features=128）：** 定义该层的输出特征数量，是设计者根据模型复杂度需求设置的超参数。较小的值（如128）可降低过拟合风险，较大的值（如512）能提升模型表达能力。

### 数学操作

全连接层执行线性变换 $output=input×{{W}^{T}}+b$ 

- 权重矩阵 $W$：形状为 (128, 9216)，共 128×9216=1,179,648 个参数
- 偏置向量 $b$：形状为 (128,)，共 128 个参数
- 总参数量：1179648+128=1179776

### 功能作用

| 功能             | 说明                                                                     |
| :--------------- | :----------------------------------------------------------------------- |
| 降维             | 将高维特征（9216维）压缩到低维空间（128维），减少后续计算量              |
| 特征融合         | 整合全局信息，将局部特征（如边缘、纹理）组合为高层语义特征（如物体部件） |
| 非线性映射的基础 | 通常后接激活函数（如ReLU），为模型引入非线性表达能力                     |



### 设计意义

**维度压缩：** 通过减少特征维度（9216 → 128），显著降低后续层的计算量（如后续分类层参数量从 $128×10=1280$ 变为 $9216×10=92160$）。

**特征抽象：** 将低级视觉特征（如卷积层提取的边缘、纹理）组合为高级语义特征（如数字形状）。

**模型容量控制：** 输出维度（128）是超参数，调整它可平衡模型复杂度：
- 过大：易过拟合，计算成本高
- 过小：限制模型表达能力



## 激活与输出

使用ReLU激活函数引入非线性。

最终通过log_softmax输出对数概率，与负对数似然损失（NLLLoss）兼容。

原理参考[激活函数ReLU的原理与应用](https://github.com/laneston/note/blob/main/01-algorithm/ReLU.md)


# 案例结果

## 输入

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/nums.jpg" width="50%"></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/numa.jpg" width="50%"></div>

## 训练5轮模型识别效果

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_nums.jpg width="50%""></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_numa.jpg" width="50%"></div>

## 训练15轮模型识别效果

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_nums15.jpg" width="50%"></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_numa15.jpg" width="50%"></div>

## 结论

- 在数字图像为正常角度的情况下，随着训练次数增加可提高识别的准确性；
- 在数字图像为非正常角度的情况下，随着训练次数增加无法明显提高识别的准确性，需要进行多次形变（矩阵旋转、转置等）后再进行识别，且需要对此图像是否为数字进行判断，以保证识别的有效性；

