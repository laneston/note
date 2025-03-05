


在 Python 中，类（Class） 是面向对象编程（OOP）的核心概念，用于定义对象的模板（属性和行为）。以下是 Python 类定义的完整解析，结合代码示例和实际应用场景说明。


# 类的基本结构

```
class ClassName:  # 类名首字母通常大写
    def __init__(self, param1, param2):  # 构造函数，初始化对象
        self.param1 = param1  # 实例属性
        self.param2 = param2

    def method(self):         # 实例方法（必须包含 self）
        return self.param1 + self.param2
```


核心要素

类名：遵循大驼峰命名法（如 MyClass）。

构造函数 __init__：在对象创建时自动调用，用于初始化属性。

self 参数：指向类的实例本身，用于访问实例属性和方法。

实例属性：通过 self.属性名 定义，每个对象独有。

实例方法：定义类的行为，第一个参数必须是 self。

# 类的使用

## 创建对象

```
obj = ClassName("Hello", "World")  # 调用 __init__ 初始化
```

## 访问属性和方法


```
print(obj.param1)          # 输出: Hello
print(obj.method())        # 输出: HelloWorld
```


# 高级特性

## 继承（Inheritance）

子类继承父类的属性和方法，并可扩展或重写：

```
class ChildClass(ClassName):  # 继承自 ClassName
    def __init__(self, param1, param2, param3):
        super().__init__(param1, param2)  # 调用父类构造函数
        self.param3 = param3

    def method(self):        # 重写父类方法
        return f"{super().method()} - {self.param3}"

child = ChildClass("A", "B", "C")
print(child.method())       # 输出: AB - C
```


<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-pyhtonlearning/ChildClass.jpg"></div>





## 类属性与实例属性

类属性：所有对象共享。

实例属性：每个对象独立。


```
class MyClass:
    class_attr = "Shared"  # 类属性

    def __init__(self, instance_attr):
        self.instance_attr = instance_attr  # 实例属性

obj1 = MyClass("Data1")

print(MyClass.class_attr)    # 输出: Shared（通过类访问）
print(obj1.class_attr)       # 输出: Shared（通过实例访问）
print(obj1.instance_attr)    # 输出: Data1
```


<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-pyhtonlearning/MyClass.jpg"></div>



## 静态方法（Static Methods）与类方法（Class Methods）


静态方法：无需访问实例或类，用 @staticmethod 修饰。

类方法：操作类属性，用 @classmethod 修饰，第一个参数是 cls。


```
class Calculator:
    @staticmethod
    def add(a, b):          # 静态方法
        return a + b

    @classmethod
    def class_info(cls):     # 类方法
        return f"This is {cls.__name__}"

print(Calculator.add(2,3))  # 输出: 5
print(Calculator.class_info())  # 输出: This is Calculator
```

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-pyhtonlearning/Calculator.jpg"></div>


## 封装与访问控制


Python 通过命名约定实现“伪私有”：

单下划线 _var：提示为内部使用（非强制）。

双下划线 __var：名称修饰（Name Mangling），避免子类属性冲突。

```
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # “私有”属性

    def deposit(self, amount):
        self.__balance += amount

account = BankAccount(100)
# print(account.__balance)      # 报错：AttributeError
print(account._BankAccount__balance)  # 强制访问：输出 100（不推荐）
```


<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-pyhtonlearning/BankAccount.jpg"></div>



# 特殊方法（Magic Methods）


通过定义特殊方法实现对象的高级行为（如运算符重载）：


```
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):    # 重载 + 运算符
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):           # 定义 print() 时的输出
        return f"({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(4, 5)
print(v1 + v2)  # 输出: (6, 8)
```

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-pyhtonlearning/Vector.jpg"></div>







