- [输入层（Input Layer）](#输入层input-layer)
- [卷积层（Convolutional Layer）](#卷积层convolutional-layer)
- [激活函数（Activation Function）](#激活函数activation-function)
  - [基础激活函数](#基础激活函数)
    - [Sigmoid](#sigmoid)
    - [Tanh（双曲正切）](#tanh双曲正切)
    - [ReLU（Rectified Linear Unit）](#relurectified-linear-unit)
  - [改进型激活函数](#改进型激活函数)
    - [Leaky ReLU](#leaky-relu)
    - [Parametric ReLU (PReLU)](#parametric-relu-prelu)
    - [ELU（Exponential Linear Unit）](#eluexponential-linear-unit)
    - [Swish](#swish)
- [池化层（Pooling Layer）](#池化层pooling-layer)
- [全连接层（Fully Connected Layer）](#全连接层fully-connected-layer)
- [输出层（Output Layer）](#输出层output-layer)
- [其他辅助组件](#其他辅助组件)



CNN 的核心架构通过 “卷积-激活-池化” 的交替堆叠，逐层提取从低级到高级的特征，最终通过全连接层输出结果。其设计巧妙结合了局部感知、权值共享、平移不变性等特性，使其成为计算机视觉领域的基石模型。

卷积神经网络（Convolutional Neural Network, CNN）是深度学习中用于处理图像、视频、语音等具有空间或时序结构数据的核心模型。其核心架构由以下几个关键组件构成：





# 输入层（Input Layer）

功能：接收原始数据（如图像的像素矩阵）。

特点：需标准化（如归一化到 [0,1] 或 [-1,1]）以加速训练。

示例：输入图像尺寸为 28×28×3（RGB三通道）。




# 卷积层（Convolutional Layer）

功能：提取局部特征（如边缘、纹理等）。

核心操作：

卷积核（Filter）：滑动窗口在输入数据上计算局部区域的加权和。

特征图（Feature Map）：每个卷积核生成一个特征图，捕捉特定模式。

关键参数：

卷积核尺寸（如 3×3、5×5）。

步长（Stride）：滑动步长（如 1 或 2）。

填充（Padding）：边缘补零（same 保持尺寸，valid 不填充）。

数学形式：

$$
output(i,j)=\sum_{m}^{}{\sum_{n}^{}{input(i+m,j+n)⋅kernel(m,n)+offset}}
$$


[卷积的基本原理请查考另一篇文档](https://github.com/laneston/note/blob/main/01-algorithm/Convolution.md)


# 激活函数（Activation Function）

激活函数的选择直接影响模型性能。ReLU及其变体（如Leaky ReLU、Swish）是现代深度学习的首选，而Sigmoid/Softmax 专用于输出层。理解其数学特性及适用场景，结合任务需求调整，是优化模型的关键步骤。


功能：引入非线性，增强模型表达能力。

常用类型：

ReLU（Rectified Linear Unit）：f(x) = max(0, x)（缓解梯度消失，计算高效）。

Sigmoid、Tanh（较少用于中间层，多用于输出层）。

Leaky ReLU、Swish（改进非线性特性）。


## 基础激活函数

### Sigmoid

$$
σ(x)=\frac{1}{1+{{e}^{-x}}}
$$


**特点：**

输出范围 (0,1)，适合二分类输出层。

梯度饱和问题：输入较大或较小时梯度接近零，导致梯度消失。

**使用场景：**

二分类输出层（如逻辑回归）。

传统神经网络（现多被ReLU替代）。

**PyTorch函数：**

torch.nn.Sigmoid()

**MATLAB**

```
% 生成x值范围
x = linspace(-5, 5, 100);

% 定义Sigmoid函数
sigmoid = @(x) 1 ./ (1 + exp(-x));

% 计算对应的y值
y = sigmoid(x);

% 绘制图像
figure;
plot(x, y, 'b', 'LineWidth', 2); % 蓝色粗线
hold on;
plot(0, 0.5, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % 标记中心点
title('Sigmoid函数');
xlabel('x');
ylabel('f(x)');
grid on;
axis([-5 5 0 1]); % 设置坐标轴范围
hold off;
```


### Tanh（双曲正切）

$$
tanh(x)=\frac{{{e}^{x}}-{{e}^{-x}}}{{{e}^{x}}+{{e}^{-x}}}
$$


特点：

输出范围 (-1, 1)，均值为零，收敛速度比Sigmoid快。

同样存在梯度饱和问题。

使用场景：

循环神经网络（RNN）的隐藏层。

需零中心化输出的场景。

**PyTorch函数：**

torch.nn.Tanh()

**MATLAB**



### ReLU（Rectified Linear Unit）

$$
ReLU(x)=max(0,x)
$$

特点：

计算高效，缓解梯度消失（正区间梯度为1）。

输出非零中心化，存在“死亡ReLU”问题（负数输入梯度为零）。

使用场景：

卷积神经网络（CNN）和全连接层的隐藏层（默认选择）。

**PyTorch函数：**

torch.nn.ReLU()

**MATLAB**

```
```

## 改进型激活函数


### Leaky ReLU


$$
LeakyReLU(x)=\left \{{\begin{matrix}x&if x>0\\αx&otherwise\end{matrix}}\right .
$$

（默认 $α=0.01$）

特点：

解决“死亡ReLU”问题，允许负数区间有微小梯度。

使用场景：

需缓解神经元死亡问题的深层网络。


**PyTorch函数：**

torch.nn.LeakyReLU(negative_slope=0.01)

**MATLAB**



### Parametric ReLU (PReLU)

特点：

类似Leaky ReLU，但 α 是可学习参数。

使用场景：

需要自适应调整负数区间的场景。

**PyTorch函数：**

torch.nn.PReLU(num_parameters=1, init=0.25)

**MATLAB**

### ELU（Exponential Linear Unit）

$$
ELU(x)\left \{{\begin{matrix}x&if x>0\\α({{e}^{x}}-1)&otherwise\end{matrix}}\right .
$$

特点：

负数区间输出平滑，接近零均值，缓解梯度消失。

计算复杂度略高（涉及指数运算）。

使用场景：

对噪声敏感的任务（如图像生成）。

**PyTorch函数：**

torch.nn.ELU(alpha=1.0)

**MATLAB**

### Swish

$$
Swish(x)=x⋅σ(βx)
$$

（β 可学习，默认 β=1）

特点：

Google提出，平滑非单调，实验表明在深层网络中优于ReLU。

使用场景：

替代ReLU用于复杂模型（如Transformer、ResNet）。

**PyTorch函数：**

```
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```











# 池化层（Pooling Layer）

功能：下采样（降维），增强平移不变性，减少计算量。

常见操作：

最大池化（Max Pooling）：取局部区域最大值（保留显著特征）。

平均池化（Average Pooling）：取局部区域平均值（平滑特征）。

参数：池化窗口大小（如 2×2）和步长（通常等于窗口大小）






# 全连接层（Fully Connected Layer）

功能：整合全局特征，输出分类或回归结果。

特点：

每个神经元与前一层的所有神经元连接。

通常位于网络末端（如分类任务中的类别概率输出）。

示例：将特征图展平后输入全连接层，输出 10 维向量（对应 10 分类）。






# 输出层（Output Layer）

功能：根据任务类型生成最终结果。

常见形式：

分类任务：Softmax 函数输出概率分布（如 [0.1, 0.7, 0.2]）。

回归任务：线性输出（如预测坐标值、价格等）。



# 其他辅助组件

批量归一化（Batch Normalization）：

加速训练，缓解梯度消失/爆炸，减少对初始化的敏感度。

Dropout：

随机丢弃部分神经元，防止过拟合。

残差连接（Residual Connection）：

解决深层网络梯度退化问题（如 ResNet 中的跳跃连接）。




以经典的 LeNet-5 为例：

```
输入层 → 卷积层 → 激活函数（ReLU） → 池化层 → 卷积层 → 激活函数 → 池化层 → 全连接层 → 输出层（Softmax）
```










