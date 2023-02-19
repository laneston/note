本篇文章将会用两个例子来说明 Makefile 是如何运行的，笔者的目的是用较小的篇幅来概括初学者在编写 Makefile 时会遇到的问题，在看完这篇文章后，也能开始写自己的 Makefile 。

- 平台: Ubuntu 20
- 工具: make, gcc

# 什么是Makefile

Makefile文件表述的是文件的依赖关系，告诉编译器怎么去编译和链接当中的文件。如果能掌握Makefile文件的编写方法，就能脱离可视化编译器，使用编译链工具编译出所需的目标文件。

## 我们开始吧

在看这篇文章时，就默认读者们已经掌握了在 Linux 系统上使用 GCC 与 make 工具的技能。如果在 Windows 上使用子系统的话，推荐用子系统的 vi 工具创建 Makefile 文件，而不是 Windows 系统上带 .mk 后缀的文件，因为这样可能会令 make 无法识别出 Makefile 文件。

其实，一个完成的编译过程包括：预处理、编译、汇编、链接。我们通常把 预处理+编译+汇编 统称为编译。

1. **预处理：** 展开头文件/宏替换/去掉注释/条件编译；
2. **编译：**   检查语法，生成汇编；
3. **汇编：**   汇编代码转换机器码；
4. **链接：**   链接到一起生成可执行程序。

我们先了解一下编译器常用的的命令：

| 选项  | 含义                                             |
| :---: | :----------------------------------------------- |
|  -v   | 查看gcc编译器的版本，显示gcc执行时的详细过程     |
|  -o   | 指定输出文件名为file，这个名称不能跟源文件名同名 |
|  -E   | 只预处理，不会编译、汇编、链接                   |
|  -S   | 只编译，不会汇编、链接                           |
|  -c   | 编译和汇编，不会链接                             |

如果我们想看编译出的中间文件，可以写入以下命令：

```
gcc -S main.c
```

## 编译和链接

编译和链接是获得目标文件的主要方式，在常见的stm32f1xx/stm32f4xx系列的编程任务中，我们可能会用到keil/MDK这个工具，当点击其中的build按键时，我们就能在输出文件夹中获得bin（二进制）或者HEX（十六进制）文件。之后，我们就可以通过MDK自带的烧录工具，使用JLink/STLink将目标文件烧录到芯片的Flash当中，抑或是通过ISP工具烧录到芯片的Flash当中。

在这个过程当中，编译器已经帮我们执行了编译和链接两个步骤。以下我会通过简单的例子说明这一个过程是如何在编译链上体现的。

创建一个名为 main.c 的文件，并写入以下代码：

```C
#include <stdio.h>

int main(int argv, char argc[])
{
    int num1=0;
    int num2=0;
    int revalue=0;

    printf("please input num 1\r\n");
    scanf("%d",&num1);

    printf("please input num 2\r\n");
    scanf("%d",&num2);

    revalue = num1 + num2;

    printf("result is ：%d\r\n", revalue);
    return 0;
}
```

我们在Ubuntu平台上用 **GCC编译器** 编译以上代码：在当前文件夹内，使用以下命令：

```
gcc main.c -o app
```

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Makefile/20200812144507.jpg"></div>

我们可以从上图所示，得到一个名为 app 的目标文件。

输入以下命令即可执行目标文件:

```
./app
```
以上过程便是我们说的编译与链接。

## 创建单目录工程

创建一个名为 Test_A 的文件夹，当然，也可以是其他名字。然后在里面创建以下文件：

**main.c**:

```
#include <stdio.h>
#include "addition.h"
#include "subtraction.h"
#include "division.h"
#include "multiplication.h"

int main(int argc, char* argv[])
{
	char symbol;
	float num_1, num_2;
	float result;

	printf("please input the symbol!\r\n");
	scanf("%c",&symbol);

	printf("please input number 1\r\n");
	scanf("%f",&num_1);

	printf("please input number 2\r\n");
	scanf("%f",&num_2);

	switch(symbol)
	{
		case '+':
			result = addition(num_1, num_2);
			break;
		case '-':
			result = subtraction(num_1, num_2);
			break;
		case '*':
			result = multiplication(num_1, num_2);
			break;
		case '/':
			result = division(num_1, num_2);
			break;
		default:
			break;		
	}

	printf("the result is:%.2f\r\n",result);

	return 0;
}
```

**addition.h**

```
#ifndef ADDITION_H
#define ADDITION_H

float addition(float a, float b);
#endif/*ADDITION_H*/
```

**addition.c**

```
float addition(float a, float b)
{
    return (a+b);
}
```

**division.h**

```
#ifndef DIVISION_H
#define DIVISION_H

float division(float a, float b);
#endif/*DIVISION_H*/
```

**division.c**

```
float division(float a, float b)
{
    return (a/b);
}
```

**multiplication.h**

```
#ifndef MULTIPLICATION_H
#define MULTIPLICATION_H

float multiplication(float a, float b);
#endif/*MULTIPLICATION_H*/
```

**multiplication.c**

```
float multiplication(float a, float b)
{
    return (a*b);
}
```

**subtraction.h**

```
#ifndef SUBTRACTION_H
#define SUBTRACTION_H

float subtraction(float a, float b);
#endif/*SUBTRACTION_H*/
```

**subtraction.cs**

```
float subtraction(float a, float b)
{
    return (a-b);
}
```

可以看出，这几个文件构成的工程是一个做2元加减乘除的运算。

## 编写Makefile文件


文件1为主函数，主函数中有4个子函数，分别是加法运算函数 addition(int a, int b) ，减法运算函数 subtraction(int a, int b)，乘法函数和除法函数。所以我们得知，要生成目标文件 app，除了需要 main.o 文件，还需要依赖 addition.o subtraction.o multiplication.o division.o 4个文件。而生成 main.o 文件需要依赖 main.c文件；而生成 subtraction.o 文件需要依赖 subtraction.c文件；而生成 addition.o 文件需要依赖 addition.c文件。如此类推，我们可以得出 Makefile 的书写方式。

在同一个文件夹中创建一个 Makefile 文件，注意的是，这个 Makefile 文件是没有任何后缀的。

**Makefile**

```
app: main.o addition.o subtraction.o multiplication.o division.o
	gcc -o app main.o addition.o subtraction.o multiplication.o division.o

main.o: main.c
	gcc -c main.c

addition.o: addition.c
	gcc -c addition.c

subtraction.o:subtraction.c
	gcc -c subtraction.c

multiplication.o: multiplication.c
    gcc -c multiplication.c

division.o: division.c
    gcc -c division.c

.PHONY : clean
clean :
	-rm main.o addition.o subtraction.o multiplication.o division.o app
```

## Makefile的编写规则

```
target... : prerequisites ...

    command
```

- **target** 是一个目标文件，可以是中间文件，也可以是执行文件，还可以是一个标签（Label）。
- **prerequisites** 就是要生成那个 **target** 所需要的文件，就是我们说的依赖文件。
- **command** 是 make 需要执行的命令（任意的Shell命令）。需要注意的是，命令要以一个Tab键作为开头。

把这一规则跟上面的 Makefile 文件对照，就明白了基本的大概，冒号(:)前面的是目标，后面是生成目标所依赖的文件，这里声明了它们之间的关系。然后下一行就是生成目标文件的命令行，这跟在控制台输入的 GCC 命令行是一样的。**值得注意的是，命令行前面必须用一个[tab]来开始。**




# Makefile的简化

当这一步，我相信大家都已经基本明白了，其实 Makefile 就是一个脚本文件。我们发现 main.o addition.o subtraction.o multiplication.o division.o 在 Makefile 文件中已经出现了3次，那我们有没有办法可以少写一些内容，让篇幅变得更加简化易读呢？答案是肯定的。

## Makefile的变量使用

其实，我们可以像 C 一样，用一个符号来代替一个字符串内容。

```
OBJ = main.o addition.o subtraction.o multiplication.o division.o

app: $(OBJ)
	gcc -o app $(OBJ)

main.o: main.c
	gcc -c main.c

addition.o: addition.c
	gcc -c addition.c

subtraction.o:subtraction.c
	gcc -c subtraction.c

multiplication.o: multiplication.c
    gcc -c multiplication.c

division.o: division.c
    gcc -c division.c

.PHONY : clean
clean :
	-rm $(OBJ) app
```

上面的例子中，我们用 OBJ 来代替了 main.o addition.o subtraction.o multiplication.o division.o 这一长串文件。注意的是，在引用变量时，需要用 $() 来包括住变量。

## Makefile的静态模式

我们不用纠结什么是静态模式，不然容易被这个名称迷惑住。现在我们脑子里就是想着怎么将这个 Makefile 文件继续化简，使得以后要往这个工程里面继续添加时，修改的内容达到最小化。乃至可以写个工程模板，即便往后继续向其中的文件夹添加文件，也无需修改 Makefile 文件。

我们观察上面的 Makefile 文件，发现当中有没有什么规律。

其实，它是可以进一步变成这样的：

```
OBJ = main.o addition.o subtraction.o multiplication.o division.o

app: $(OBJ)
	gcc -o app $(OBJ)

%.o:%.c
	gcc -c $< -o $@

.PHONY : clean
clean :
	-rm $(OBJ) app
```

我们发现，里面的 .c 文件都不见了，以后我们增删文件只需修改第一行就行，让我们编写 Makefile 的时间大大减少了。我们又是怎么做到的呢？因为我们用了静态模式。有朋友会问，这个怎么该理解呢？好的，那我现在细细给大家解析。

main.o 文件依赖的是 main.c 文件，以及文件里面包含的 addition.h subtraction.h multiplication.h division.h 头文件；addition.o 文件依赖的是 addition.c 文件。如此类推，我们发现 $(OBJ) 中每个文件的编译方式都是一个套路，就是：*.o:*.c

所以我们可以用： %.o:%.c 来代替5个 .o 文件的依赖声明。

那 gcc -c $< -o $@ 又是怎么回事呢？有朋友又问了。这个东西说来话长，以下请听我细细道来。


## 通配符

- $@：代表的是目标文件
- $^：代表的是内容中所有的依赖文件
- $<：代表的是内容中第一个依赖文件
- $?：构造所需文件列表中更新过的文件

- =  最基本的赋值
- := 覆盖之前的值
- ?= 如果没有被赋值过就赋予等号后面的值
- += 追加等号后面的值

各位看官，$< 代表的是内容中第一个依赖文件。在 main.o: main.c 一行中说的就是 main.c ；$@ 代表的是目标文件，指的就是冒号前的 main.o 文件。

整个命令看来，它就是说：由第一个依赖文件通过编译(gcc -c $<)生成目格式为 .o 的文件(-o $@) 

那有杠友会问，那为什么不写成 gcc -c $^ -o $@

听到这句话，我反手就是一个……赞，这个问题问得太好了。确实，冒号后面的就是1个 main.c 文件，且在用在这个例子中也能运行通过的。$^ 代表的是内容中所有的依赖文件，有些时候冒号后跟的不是1个文件时就会把所有文件都加进来，然后就产生错误了，但绝大部分情况下，一个 .o 文件依赖一个 .c 文件，相应的头文件也会由 .c 文件内部包括，我们不必要为特殊情况而写一个不通用的 Makefile。

看到这里，很多朋友都应该大呼厉害，当然指的是设计规则的人。又有人会问，能不能把 OBJ = main.o addition.o subtraction.o multiplication.o division.o 这一串东西也去掉啊，让 make 自己在文件夹里找依赖文件，这样我们以后往文件夹里添加文件也不用修改任何东西了。听到这句话，我又是反手就是一个……赞。这个当然可以啦！

## 函数

我们可以这样写：

```
SRC = $(wildcard *.c)
OBJ = $(SRC:%.c=%.o)

app: $(OBJ)
	gcc -o app $(OBJ)

%.o:%.c
	gcc -c $^ -o $@

.PHONY : clean
clean :
	-rm $(OBJ) app
```

满意了吧，没有一个 .o 或 .c 文件需要我们手动添加了。这时朋友们都热切想知道到底是怎么做到的。

- $(subst from, to, text)  把字串 text 中的 from 字符串替换成 to。
- $(patsubst pattern, replacement, text)  查找 text 中的单词（单词以“空格”、“Tab”或“回车”“换行”分隔）是否符合模式 pattern ，如果匹配的话，则以 replacement 替换。这里， pattern 可以包括通配符“%”，表示任意长度的字串。如果  replacement 中也包含“%”，那么， replacement 中的这个“%”将是 pattern 中的那个“%”所代表的字串。（可以用“\”来转义，以“\%”来表示真实含义的“%”字符）
- $(notdir names)  从文件名序列 names 中取出非目录部分。非目录部分是指最后一个反斜杠（“/”）之后的部分。
- $(wildcard PATTERN) 获取匹配 PATTERN 的所有对象。

这里，我们用到的是通配符函数。函数的通用格式为：$(function [param]) 这跟我们的变量用法很像吧。

1. 第1行中，将所有 .c 后缀的文件放入变量 SRC 中
2. 然后，将 SRC 中的 .c 文件变为 .o 并放入变量 OBJ 中

剩下的就不用我说了。那有杠友又要问了，你第二步就把 .c 变成 .o 了，那是不是不用写 %.o:%.c 了。是的，这里可以不用写了，可以变为：

```
SRC = $(wildcard *.c)
OBJ = $(SRC:%.c=%.o)

app: $(OBJ)
	gcc -o app $(OBJ)

.PHONY : clean
clean :
	-rm $(OBJ) app
```

我们也可以写成：

```
OBJ := $(patsubst %.c,%.o,$(wildcard *.c))

app: $(OBJ)
	gcc -o app $(OBJ)

.PHONY : clean
clean :
	-rm $(OBJ) app
```

或：

```
OBJ := $(patsubst %.c,%.o,$(wildcard *.c))

app:  $(OBJ)
	gcc -o $@ $^
```

值得注意的是，上面的命令行中，我们用到了的是：

```
    gcc -o $@ $^
```
而不是：
```
    gcc -o $@ $<
```
虽然按常规的理解，$< 代表的是第一个依赖的文件，而目标文件 app 后只接了一个变量 $(objects)，按照某些人的理解,$(objects) 需看作是一个整体，这个“依赖文件”理应是变量 $(objects)，遗憾的是，里面包含了：main.o addition.o subtraction.o division.o multiplication.o 几个文件。如果用符号 $< 指代的是第一个依赖文件 main.o 所以不能用 $< 

## 清空编译文件

我们发现每次编译的过程中都会产生一大堆中间文件，这令人感到杂乱，每个 Makefile 中都应该写一个清空目标文件（.o和执行文件）的规则，这不仅便于重编译，也很利于保持文件的清洁。

```
.PHONY : clean
clean :
	-rm $(OBJ) app
```

.PHONY 意思表示 clean 是一个“伪目标”，在rm命令前面加了一个小减号的意思就是，也许某些文件出现问题，但不要管，继续做后面的事。clean 的规则不要放在文件的开头，不然，这就会变成make 的默认目标，一般我们都会放在文件结尾处。

输入命令行： make clean 就可以清除对应的文件了。

到这里为止，单目录里的 Makefile 我们就已经基本学完了。那又有朋友问了，我的工程很大啊，不是一个文件夹能装得下的，多目录下又该怎么写呢？

不要急，我亲爱的朋友。

# 多目录下的Makefile

如果有以下结构的工程，我们又该怎么写 Makefile 呢？

```
TEST_B
	|__generalOperation
	|	|__addition.h
	|	|__addition.c
	|	|__subtraction.h
	|	|__subtraction.c
	|
	|__shiftOperation
	|	|__division.h
	|	|__division.c
	|	|__multiplication.h
	|	|__multiplication.c
	|
	|__User
	|	|__main.c
	|	|__main.h
	|
	|__Makefile
```

那简单，我们用变量把文件路径包括进来就行。

```
SRC += ./User/*.c
SRC += ./User/*.c
SRC += ./generalOperation/*.c
```

这部分是将3个文件夹中的 .c 文件追加到变量 SRC 中。但值得注意的是，这样子追加的变量是不能直接用的，因为 SRC 中保存的是原始字符串 \./User/*.c \./User/*.c \./generalOperation/*.c

需要用 RAW_SRC := $(wildcard $(SRC)) 来转换一下。保存到 RAW_SRC 中的就是每个文件夹中的 .c 文件的路径啦。

至于头文件呢，我们就需要用到 -I 这个符号了，后面可接头文件的保存路径。

```
DIR_INC += -I./User
DIR_INC += -I./shiftOperation
DIR_INC += -I./generalOperation
```

综上所述，我们可以得出 Makefile 文件为：

```
SRC += ./User/*.c
SRC += ./shiftOperation/*.c
SRC += ./generalOperation/*.c

DIR_INC += -I./User
DIR_INC += -I./shiftOperation
DIR_INC += -I./generalOperation

RAW_SRC := $(wildcard $(SRC))

OBJ := $(RAW_SRC:%.c=%.o)

app: $(OBJ)
	@echo $(OBJ)
	gcc -o $@ $(OBJ)

%.o:%.c
	gcc -c $(DIR_INC) $< -o $@

.PHONY : clean
clean :
	-rm app
	    ./User/*.o
        ./shiftOperation/*.o
        ./generalOperation/*.o
```

这时候，有朋友就不能满意了。因为发现 .o 文件跟 .c 混在一起了，目标文件也跟 Makefile 文件放在一起了。之所以用多个文件夹区分文件，是因为想把不同的文件归类啊。

那我们就创建一个 output 文件夹吧。

```
SRC += ./User/*.c
SRC += ./shiftOperation/*.c
SRC += ./generalOperation/*.c

DIR_INC += -I./User
DIR_INC += -I./shiftOperation
DIR_INC += -I./generalOperation

RAW_SRC := $(wildcard $(SRC))

OBJ := $(RAW_SRC:%.c=%.o)

./output/app: $(OBJ)
	@echo $(OBJ)
	gcc -o $@ $(OBJ)
	-mv ./User/*.o ./output
	-mv ./shiftOperation/*.o ./output
	-mv ./generalOperation/*.o ./output

%.o:%.c
	gcc -c $(DIR_INC) $< -o $@

.PHONY : clean
clean :
	-rm ./output/*.o\
	    ./output/app
```

有朋友看到这段 Makefile 可能就惊叹了，什么？竟然直接用 mv 命令来把 .o 文件剪切到 output 文件夹。没错，就是这么直接，因为 Makefile 其实就是 Shell 脚本啊。其实这里还可以变得更巧妙些，在 .o 文件编译时就可以将文件移动到 output 文件夹中。

```
SRC += ./User/*.c
SRC += ./shiftOperation/*.c
SRC += ./generalOperation/*.c

DIR_INC += -I./User
DIR_INC += -I./shiftOperation
DIR_INC += -I./generalOperation

RAW_SRC := $(wildcard $(SRC))

OBJ := $(RAW_SRC:%.c=%.o)

./output/app: $(OBJ)
	@echo $(OBJ)
	gcc -o $@ $(OBJ)

%.o:%.c
	gcc -c $(DIR_INC) $< -o $@
	-mv $@ ./output

.PHONY : clean
clean :
	-rm ./output/*.o\
	    ./output/app
```
当然， Makefile 的运用还可以更巧妙。 也有 vpath 这种关键词来辅助文件路径的寻找。但这篇文章的目的就是用最直接简单有效的方式教会大家编写 Makefile ，而进阶的事情，还是留到之后的实践再慢慢学习吧。