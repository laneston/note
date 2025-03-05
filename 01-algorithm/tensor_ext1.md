







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


## super() 的作用

Python 2：需显式传递子类名和实例，即 super(子类名, self)。

Python 3：可简化为 super().__init__()，解释器会自动推断。
示例中的写法 super(DigitRecognizer, self).__init__() 是为了兼容旧版本，但现代PyTorch代码通常直接写 super().__init__()。














# 网络结构详解

## 卷积层1 (conv1)

输入：1通道（灰度图）

输出：32通道，使用3x3卷积核，步长1

作用：检测初级特征（如边缘、线条）。

## 卷积层2 (conv2)

输入：32通道

输出：64通道，3x3卷积核，步长1

作用：捕获更复杂的特征（如曲线、组合形状）。

## 最大池化

窗口大小2x2，将特征图尺寸减半（如24x24 → 12x12），保留显著特征并减少计算量。

## Dropout正则化

dropout1（25%丢弃率）：在池化后应用，防止过拟合。

dropout2（50%丢弃率）：在全连接层前应用，增强泛化能力。

## 全连接层 (fc1和fc2)

fc1：将展平后的9216维特征（64通道×12×12）映射到128维，进一步整合特征。

fc2：输出10维，对应0-9分类结果。

## 激活与输出

使用ReLU激活函数引入非线性。

最终通过log_softmax输出对数概率，与负对数似然损失（NLLLoss）兼容。







