

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

# 张量运算

```
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 加法
add = a + b  # 或 torch.add(a, b)
print("\n加法结果:\n", add)

# 乘法（逐元素）
mul = a * b
print("\n逐元素乘法:\n", mul)

# 矩阵乘法
matmul = a @ b.T  # 或 torch.matmul(a, b.T)
print("\n矩阵乘法结果:\n", matmul)

# 广播机制
c = torch.tensor([10, 20])
broadcast = a + c  # c被广播为[[10,20], [10,20]]
print("\n广播加法:\n", broadcast)
```

# GPU操作

```
# 将张量移动到GPU
if torch.cuda.is_available():
    a_cpu = torch.tensor([[1, 2], [3, 4]])
    a_gpu = a_cpu.to('cuda')  # 或 a_cpu.cuda()
    print("\nGPU张量:\n", a_gpu)

    # 在GPU上运算
    b_gpu = torch.ones(2, 2, device='cuda')
    result_gpu = a_gpu + b_gpu
    print("\nGPU运算结果:\n", result_gpu)
```

# 与NumPy互操作

```
import numpy as np

# 张量 → NumPy数组
a_torch = torch.tensor([[1, 2], [3, 4]])
a_np = a_torch.numpy()
print("\nTorch转NumPy:\n", a_np)

# NumPy数组 → 张量
b_np = np.array([[5, 6], [7, 8]])
b_torch = torch.from_numpy(b_np)
print("\nNumPy转Torch:\n", b_torch)

# 注意：数据共享，修改一个会影响另一个
a_torch.add_(1)
print("\n修改后Torch张量:\n", a_torch)
print("对应的NumPy数组:\n", a_np)  # a_np也会被修改
```

# 自动求导

```
# 创建需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)

# 定义计算图
y = x ** 2 + 2 * x + 3

# 反向传播计算梯度
y.backward()

# 查看x的梯度
print("\nx的梯度:", x.grad)  # 输出: 2*2 + 2 = 6.0

# 复杂例子（矩阵运算）
A = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[5.], [6.]])
output = (A @ b).sum()  # 计算总和以便标量反向传播
output.backward()
print("\nA的梯度:\n", A.grad)  # 梯度为b的转置重复两次
```



