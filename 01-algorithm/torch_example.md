

以下是PyTorch张量的详细示例，涵盖创建、属性、运算、GPU操作、与NumPy的互操作及自动求导功能：


# 创建张量

```
import torch

# 从列表创建
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("从列表创建:\n", a)

# 全零张量
b = torch.zeros(2, 3)
print("\n全零张量:\n", b)

# 全一张量
c = torch.ones(2, 3)
print("\n全一张量:\n", c)

# 随机张量（正态分布）
d = torch.randn(2, 3)
print("\n随机张量:\n", d)

# 指定数据类型和设备
e = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
print("\nGPU上的张量:\n", e)
```



