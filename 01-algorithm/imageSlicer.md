

这段代码是一个名为`process_image_to_mnist`的函数，它的目标是将包含手写数字的图片切割成单个数字，并将其转换为MNIST数据集格式。

```
def process_image_to_mnist(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path")
    
    # 图像预处理流程
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.DAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 轮廓检测与处理
    contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        
        # 高级过滤条件（可根据实际调整）
        if area > 80 and 0.25 < aspect_ratio < 4:
            valid_contours.append(cnt)
    
    # 按从左到右排序数字
    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    digits = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = processed[y:y+h, x:x+w]
        
        # 专业MNIST格式化处理
        # 步骤1：创建正方形画布并保持宽高比
        (h, w) = roi.shape
        padding = 4
        if w > h:
            pad_total = w - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_total = h - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            roi = cv2.copyMakeBorder(roi, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        
        # 步骤2：添加额外padding并缩放
        roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
        
        # 步骤3：居中放置到28x28画布
        mnist_format = np.zeros((28, 28), dtype=np.uint8)
        mnist_format[4:24, 4:24] = roi  # 居中放置
        
        digits.append(mnist_format)
    
    # PyTorch标准化处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST官方参数
    ])
    
    return [transform(digit) for digit in digits]
```

# 函数定义与图像读取

```
def process_image_to_mnist(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path")
```

作用：定义处理函数并加载图像

关键点：

- cv2.imread()返回BGR格式的NumPy数组
- 空值检查防止无效路径导致后续崩溃

注意：OpenCV不支持中文路径，需确保路径正确

# 图像预处理流水线

```
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
```

处理流程：

## cv2.cvtColor

1. 颜色空间转换：将三通道（B、G、R）的彩色图像转换为单通道的灰度图像。
2. 数据简化：减少计算量和内存占用，适用于不需要颜色信息的任务（如边缘检测、特征提取）。
3. 标准化处理：统一输入格式，便于后续算法处理。


转换原理

加权平均法
   
灰度值通过以下公式计算：

$$
Gray=0.114×B+0.587×G+0.299×R
$$

权重依据：基于人眼对不同颜色光的敏感度（绿色感知最强，红色次之，蓝色最弱）。

通道顺序：由于 OpenCV 默认使用 BGR 格式，需按 B（第一通道）、G（第二通道）、R（第三通道）的顺序取值。

实现步骤
   
1. 遍历像素：对图像的每个像素分别提取 B、G、R 值。
2. 加权计算：按公式计算灰度值，结果范围为 [0, 255]。
3. 生成灰度图：将所有像素的灰度值组合成单通道图像。


| 转换类型   | 代码               | 典型应用场景             |
| :--------- | :----------------- | :----------------------- |
| BGR → 灰度 | cv2.COLOR_BGR2GRAY | 边缘检测、人脸识别       |
| BGR → HSV  | cv2.COLOR_BGR2HSV  | 颜色分割、目标追踪       |
| BGR → RGB  | cv2.COLOR_BGR2RGB  | 图像显示（Matplotlib）   |
| 灰度 → BGR | cv2.COLOR_GRAY2BGR | 灰度图上叠加彩色信息     |
| BGR → YUV  | cv2.COLOR_BGR2YUV  | 视频压缩（如 H.264）     |
| BGR → Lab  | cv2.COLOR_BGR2Lab  | 颜色差异分析（图像增强） |


## cv2.GaussianBlur

```
cv2.GaussianBlur(src, ksize, sigmaX[, sigmaY[, borderType]])
```


| 参数       | 类型      | 作用                       | 注意事项                                          |
| :--------- | :-------- | :------------------------- | :------------------------------------------------ |
| src        | 输入图像  | 需为单通道或多通道图像     | 支持8/16/32位整型或浮点型                         |
| ksize      | 核尺寸    | 高斯核的宽高（必须为奇数） | 如(5,5)表示5×5核                                  |
| sigmaX     | X轴标准差 | 控制水平方向的模糊强度     | 若为0，则自动计算为 0.3*((ksize-1)*0.5 - 1) + 0.8 |
| sigmaY     | Y轴标准差 | 控制垂直方向的模糊强度     | 默认与 sigmaX 相同                                |
| borderType | 边界处理  | 处理图像边缘的策略         | 常用 cv2.BORDER_DEFAULT                           |

高斯核（Gaussian Kernel）

定义： 核内每个像素的权重由二维高斯函数计算：

$$
G(x,y)=\frac{1}{2π{{σ}^{2}}}{{e}^{-\frac{{{x}^{2}}+{{y}^{2}}}{2{{σ}^{2}}}}}
$$

其中 $(x,y)$ 是像素到核中心的距离，$σ$ 是标准差。

特性：

- 权重呈钟形分布，中心权重最大
- 权重值随距离中心衰减
- 核尺寸越大，模糊效果越强

卷积操作

将高斯核与图像进行卷积运算：

$$
{{I}^{'}}(x,y)=\sum_{i=-k}^{k}{\sum_{j=-k}^{k}{G(i⋅j)⋅I(x+i,y+j)}}
$$

每个像素的新值是邻域像素的加权平均，权重由高斯核决定。



自适应阈值：

- 方法：高斯加权平均（ADAPTIVE_THRESH_GAUSSIAN_C）
- 模式：反色二值化（THRESH_BINARY_INV）
- 参数：11x11邻域，常数C=2

效果：生成黑白分明的二值图像

# 形态学优化

```
    # 创建椭圆核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 执行闭运算
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
```

目的：改善数字连通性

闭运算（膨胀后腐蚀）：

- 消除数字内部小孔洞
- 连接邻近的断裂笔画

椭圆核：比矩形核更自然的形状，适合手写体


## cv2.getStructuringElement

cv2.getStructuringElement 是 OpenCV 中用于**生成形态学操作所需结构元素**的核心函数，其解析如下：


### ​功能与作用

结构元素（Kernel）是形态学操作（如腐蚀、膨胀、开运算、闭运算等）的关键工具，用于定义操作中的邻域形状和大小。不同的结构元素形状会影响图像处理的效果，例如去除噪声、连接断裂区域或分离物体。

### ​函数参数解析

​**shape**：结构元素的形状，支持三种预定义类型：

- cv2.MORPH_RECT：矩形，所有元素为1，适用于一般腐蚀/膨胀操作。
- cv2.MORPH_ELLIPSE：椭圆形，适合平滑边缘或处理圆形物体。
- cv2.MORPH_CROSS：十字形，常用于连接相邻像素或细长区域处理。

​**ksize**：结构元素的大小，需为奇数的二元组 (width, height)，例如 (5,5)。奇数值确保锚点位于中心，保证对称性。

​**anchor**：锚点位置，默认为 (-1, -1) 表示中心点。用户可自定义偏移，例如 (0,0) 表示左上角，但实际使用中通常无需修改。

### ​返回值

返回一个 numpy.ndarray 类型的二值矩阵，元素值为0或1。例如：

- 矩形结构元素​（5x5）：全1矩阵。
- ​椭圆形结构元素​（5x5）：中心为椭圆形的1值分布，边缘为0。
- ​十字形结构元素​（5x5）：水平和垂直中线为1，其余为0。

## cv2.morphologyEx

cv2.morphologyEx() 是 OpenCV 中用于执行高级形态学操作的函数，结合了基础的腐蚀和膨胀操作，适用于多种图像处理场景。以下是该函数的详细解析：

```
cv2.morphologyEx(
    src,          # 输入图像（二值化或灰度图像）
    op,           # 形态学操作类型标识符
    kernel,       # 结构元素（内核）
    dst=None,     # 输出图像
    anchor=(-1,-1),  # 结构元素锚点，默认中心
    iterations=1,    # 迭代次数
    borderType=cv2.BORDER_CONSTANT,  # 边界处理方式
    borderValue=0    # 边界填充值
)
```

参数解析

​**src**​ 输入图像，需为单通道（灰度）或二值图像。通常需先进行预处理（如二值化）以提高效果。

​**op**​

形态学操作类型，可选以下常量：

- ​**cv2.MORPH_OPEN**：开运算（先腐蚀后膨胀）。用于去除小噪声、分离物体。
- ​**cv2.MORPH_CLOSE**：闭运算（先膨胀后腐蚀）。用于填补小孔、连接断裂区域。
- ​**cv2.MORPH_GRADIENT**：形态学梯度（膨胀图 - 腐蚀图）。提取物体边缘。
- ​**cv2.MORPH_TOPHAT**：顶帽变换（原图 - 开运算结果）。突出亮于背景的小区域。
- ​**cv2.MORPH_BLACKHAT**：黑帽变换（闭运算结果 - 原图）。突出暗于背景的小区域。

​**kernel**​

结构元素，决定操作效果。常用 cv2.getStructuringElement() 生成，支持形状（矩形、椭圆、十字形）和尺寸自定义。例如：`kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  # 5x5矩形内核`

**​其他参数​**

iterations：操作重复次数，次数过多可能导致图像失真。
borderType 和 borderValue：处理图像边界的填充方式，通常使用默认值即可。



# 轮廓处理

```
    # 检测外轮廓
    contours, _ = cv2.findContours(
        processed.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 轮廓过滤
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        
        if area > 80 and 0.25 < aspect_ratio < 4:
            valid_contours.append(cnt)
    
    # 按x坐标排序
    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
```

轮廓检测：

- RETR_EXTERNAL：仅检测最外层轮廓
- CHAIN_APPROX_SIMPLE：压缩冗余轮廓点

过滤条件：

- 面积>80像素：过滤噪点
- 宽高比0.25-4：排除线状或过扁图形
  
排序：确保数字按从左到右顺序处理



## cv2.findContours

cv2.findContours() 是 OpenCV 中用于检测图像轮廓的核心函数，适用于对象分割、形状分析等任务。以下是对其功能的详细解析及使用要点：


### 函数功能与参数解析

​基本作用​

在二值图像中检测轮廓，返回轮廓点集及层次结构信息。轮廓是连接相同颜色或强度的连续点构成的曲线，常用于对象边界识别。

​参数详解​`contours, hierarchy = cv2.findContours(image, mode, method[, offset])`

​**image**​

输入图像必须是二值图像（如通过 cv2.threshold() 或 cv2.Canny() 处理后的结果）。非零像素视为前景（如255），零像素为背景。

注意：原图可能被函数修改，建议传入副本。

​**mode**​（轮廓检索模式）

- cv2.RETR_EXTERNAL：仅检测最外层轮廓。
- cv2.RETR_LIST：检测所有轮廓，无层次关系。
- cv2.RETR_TREE：检测所有轮廓并构建完整的层次结构（如嵌套轮廓）。

​**method**​（轮廓近似方法）

- cv2.CHAIN_APPROX_NONE：存储所有轮廓点（高精度）。
- cv2.CHAIN_APPROX_SIMPLE：压缩冗余点（如直线仅保留端点，节省内存）。
- cv2.CHAIN_APPROX_TC89_*：Teh-Chin 链近似算法的高级压缩。

​**offset**​（可选）

轮廓点坐标的偏移量，常用于 ROI（感兴趣区域）分析。

### 返回值解析


**contours**​

类型：list，每个元素为一个轮廓的点集（numpy.ndarray），表示轮廓上的坐标点。

示例：contours[0] 表示第一个轮廓的所有点坐标。

​**hierarchy**​

类型：numpy.ndarray，描述轮廓间的层次关系。
每行包含 4 个整数，表示当前轮廓的 ​下一轮廓索引、前一轮廓索引、第一个子轮廓索引​ 和 ​父轮廓索引​（若无则为-1）。
​注意版本差异​

​OpenCV 3.x 及以上版本：返回值为 (image, contours, hierarchy)，需接收三个变量。
​旧版本：可能仅返回 (contours, hierarchy)，需根据实际版本调整代码。



## cv2.boundingRect

cv2.boundingRect() 是 OpenCV 中用于计算轮廓或点集最小外接矩形的关键函数，其功能和应用场景可通过以下解析全面理解：


### 核心功能

该函数通过计算二维点集的最小正外接矩形（与图像坐标系轴对齐），返回一个包含左上角坐标、宽度和高度的矩形参数。其核心作用是快速定位目标区域，适用于物体检测、图像分割等场景。

​输入要求：点集需以 `std::vector` 或 `Mat` 形式存储，通常来源于图像轮廓（如 cv2.findContours() 的输出）。

​数学原理：遍历点集找到最小和最大的 x、y 值，生成矩形边界框，满足：

$$
x_min = min(points[:, 0]), y_min = min(points[:, 1])
$$
$$
width = max(points[:, 0]) - x_min
$$
$$
height = max(points[:, 1]) - y_min
$$

### 参数与返回值

​参数：points（必需）

支持轮廓点集（如 contours[0]）或任意二维点集，需为单通道或多通道数组，但非二值图时会自动提取有效点。

​返回值：(x, y, w, h)

x, y：矩形左上角在图像坐标系中的坐标（原点为左上角）
w, h：矩形的宽度和高度，单位为像素


## cv2.contourArea

cv2.contourArea() 是 OpenCV 中用于计算轮廓面积的核心函数，其功能解析和注意事项如下：

### 函数原型与参数

`cv2.contourArea(contour, oriented_area=False)`

​参数解析：

- contour：轮廓的点集，通常通过 cv2.findContours() 获取。要求轮廓点至少包含 2 个点。
- oriented_area（可选）：
    - 默认 False，返回轮廓面积的绝对值。
    - 若为 True，返回带符号的面积，符号表示轮廓点方向（顺时针为负，逆时针为正）。

### 返回值

​面积计算原理：

基于格林公式，将轮廓视为多边形，通过顶点坐标的积分计算面积。

$$
S=\frac{1}{2}\oint_{L}^{​}{xdy-ydx}
$$

​方向影响：若轮廓点按顺时针排列，返回负值；逆时针则为正值。


# MNIST格式化处理

## 创建正方形画布

```
        (h, w) = roi.shape
        padding = 4
        if w > h:
            pad_total = w - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_total = h - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            roi = cv2.copyMakeBorder(roi, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
```

逻辑：

- 当宽度>高度：上下对称填充使成为正方形
- 当高度>=宽度：左右对称填充

示例：

- 原始尺寸：40x30 → 上下各加5px → 40x40
- 填充值：0（黑色背景）



cv2.copyMakeBorder 是 OpenCV 中用于为图像添加边框的核心函数，广泛应用于图像预处理、卷积操作边界处理等场景。以下是该函数的详细解析：

```
dst = cv2.copyMakeBorder(
    src,                # 输入图像（支持多通道）
    top,                # 顶部边框宽度（像素）
    bottom,             # 底部边框宽度
    left,               # 左侧边框宽度
    right,              # 右侧边框宽度
    borderType,         # 边框类型（见下方选项）
    value=None          # 当边框类型为BORDER_CONSTANT时的填充值
)
```



| 参数名      | 类型    | 必填 | 说明                                                           |
| :---------- | :------ | :--- | :------------------------------------------------------------- |
| ​src        | ​	Mat   | 是   | 输入图像，支持灰度图（单通道）或彩色图（如BGR三通道）。        |
| ​top        | ​	int   | 是   | 顶部添加的边框高度（单位：像素）。                             |
| ​bottom     | ​	int   | 是   | 底部添加的边框高度。                                           |
| ​left       | ​	int   | 是   | 左侧添加的边框宽度。                                           |
| ​right      | ​	int   | 是   | 右侧添加的边框宽度。                                           |
| ​borderType | ​	int   | 是   | 边框填充类型，可选以下常量：                                   |
|             |         |      | - cv2.BORDER_CONSTANT：常数填充                                |
|             |         |      | - cv2.BORDER_REPLICATE：复制边缘像素                           |
|             |         |      | - cv2.BORDER_REFLECT：镜像反射                                 |
|             |         |      | - cv2.BORDER_REFLECT_101：带边缘的镜像反射                     |
|             |         |      | - cv2.BORDER_WRAP：平铺填充                                    |
| ​value      | ​Scalar | 否   | 当borderType=cv2.BORDER_CONSTANT时，指定填充颜色（默认黑色）。 |
|             |         |      | - 灰度图：单个值（如0表示黑色）                                |
|             |         |      | - 彩色图：BGR三元组（如(0, 255, 0)表示绿色）                   |


返回值

​dst：输出图像（Mat类型），尺寸为 (src.rows + top + bottom, src.cols + left + right)，通道数与输入图像一致。




## 缩放与二次填充


```
        roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
```

二次填充：四周各加4px，形成28x28画布

缩放：使用区域插值（INTER_AREA）保持边缘清晰

## 参数详解

`cv2.resize` 是 OpenCV 中用于调整图像尺寸的核心函数，支持多种缩放方式和插值算法。以下是对其功能、参数、原理及使用场景的详细解析：

核心功能

​图像缩放​

- ​放大/缩小：通过指定目标尺寸（dsize）或缩放因子（fx, fy）调整图像大小。当目标尺寸小于原图时缩小，反之则放大。
- ​纵横比保持：默认保持原图宽高比。若需改变比例，需显式指定非等比参数。

​插值算法​

根据不同的缩放需求选择合适的插值方法：

- ​**INTER_NEAREST**：最近邻插值，速度快但易产生锯齿（适合实时性要求高场景）。
- **INTER_LINEAR**：双线性插值（默认），平衡速度与质量（适用于一般缩放）。
- **INTER_CUBIC**：双三次插值，通过 4x4 邻域计算，适合放大图像（细节更平滑）。
- **INTER_AREA**：基于像素区域重采样，适合缩小图像（减少锯齿）。
- **INTER_LANCZOS4**：8x8 邻域插值，高质量但计算量大（用于高精度需求）。

```
cv2.resize(src, dsize, fx=None, fy=None, interpolation=INTER_LINEAR)
```

- **src**：输入图像（numpy.ndarray 格式）。
- ​**dsize**：目标尺寸元组 (width, height)。若设为 None，则需指定 fx 和 fy。
- ​**fx/fy**：沿 x/y 轴的缩放因子。例如 fx=0.5 表示宽度缩小一半。
- ​**interpolation**：插值方法（默认为双线性插值）。


*注意：dsize 优先级高于 fx/fy。若同时指定，dsize 生效，fx/fy 被忽略。*


## 原理与实现

​坐标变换​

函数通过几何变换将目标像素位置映射到原图坐标。例如，原图尺寸为 (w1, h1)，目标尺寸为 (w2, h2)，则目标像素 (x2, y2) 对应原图坐标为：$x1 = x2 * (w1/w2)$ , $y1 = y2 * (h1/h2)$

最终通过插值算法计算目标像素值。

​性能优化​

​速度：INTER_NEAREST 最快，INTER_LANCZOS4 最慢。

​内存管理：放大高分辨率图像时需注意内存消耗，建议结合 GPU 加速（如 cv2.cuda.resize）。



## 居中放置

```
        mnist_format = np.zeros((28, 28), dtype=np.uint8)
        mnist_format[4:24, 4:24] = roi
```

数学验证：

- 20x20 ROI → 放入28x28画布
- 起始位置：(4,4) → 结束位置：(24,24)

结果：数字精确居中显示

## PyTorch标准化

```
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return [transform(digit) for digit in digits]
```

处理流程：

- ToTensor()：转换维度为C×H×W，并归一化到[0,1]
- Normalize：应用MNIST数据集统计值

输出格式：每个元素为形状(1,28,28)的标准化张量


**参数调整指南**

| 参数             | 推荐值      | 作用             | 调整影响             |
| :--------------- | :---------- | :--------------- | :------------------- |
| 高斯核大小       | (5,5)       | 降噪强度         | 增大值更模糊         |
| 自适应阈值块大小 | 11          | 局部计算范围     | 增大适应更大光照变化 |
| 形态学核大小     | (3,3)       | 连接强度         | 增大可连接更远区域   |
| 最小面积         | 80	过滤噪点 | 根据数字大小调整 |
| 宽高比范围       | 0.25-4      | 形状过滤         | 宽松范围适应不同字体 |



transforms.Compose 是深度学习框架中用于组合多个数据预处理或增强操作的核心函数，其设计理念在 PyTorch 和 MindSpore 等框架中均有体现。以下从功能解析、参数说明、使用场景和框架差异四个维度进行详细解析：

## 操作链式组合​

Compose 将多个独立的数据变换操作（如裁剪、归一化）按顺序封装为一个整体，形成可调用的处理流水线。例如，在图像处理中，典型流程可能包含 Resize → ToTensor → Normalize，这些操作会被整合为一个函数对象。

## ​执行顺序控制​

变换列表中的操作按声明顺序依次执行。例如，若先执行 ToTensor 将图像转为张量，后续的 Normalize 才能对张量进行数值标准化。顺序错误可能导致类型不匹配（如对 PIL 图像直接应用数值运算）。


# 测试结果

## 输入

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/nums.jpg"></div>

## 输出

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digitalSlice.jpg"></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digitalSensor.jpg"></div>
