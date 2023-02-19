文章围绕makefile文件的编写方式，向读者讲述如何在ubuntu平台上用交叉编译链 arm-none-eabi- 编译出 STM32F4xx 系列 MCU 的执行文件。文章核心在于讲述 arm-none-eabi- 在 Makefile 中的应用过程，对比于嵌入式可视编译器 keil_v5 有什么共同点，编译思维是怎样的，并完成一个简单项目 <a href="https://github.com/laneston/STM32F4xx_LED-Makefile"> STM32F4xx_LED-Makefile </a> 的编译工作。 如果还没对 Makefile 入门的朋友可以查看我的另一篇文章 <a href="https://github.com/laneston/Note/blob/master/Hey_Makefile.md"> Hey Makefile! </a>，它会帮你快速上手学会写 Makefile 。

- 平台： Ubuntu20，STM32F407ZGT6
- 工具： arm-none-eabi-

# 初见交叉编译链

为什么要使用交叉编译链工具呢？在嵌入式开发过程中有宿主机和目标机的角色之分：宿主机是执行编译、链接嵌入式软件的计算机；目标机是运行嵌入式软件的硬件平台。简单地说，就是在一个平台上生成另一个平台上的可执行代码。

有朋友可能会问了：为什么要在一个平台上编译另一个平台的执行文件呢？不能像 PC 那样在 PC 上编译 PC 能执行的文件呢？嗯……你觉得能在 STM32 上安装一个编译工具，然后接上鼠标键盘和屏幕，编译出一个LED流水灯程序来吗？你有问过它的 Flash 和 RAM 够用吗？

说到这里，我们该用哪个交叉编译链工具呢？

**<a href = "https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads">gcc-arm-none-eabi</a>** 

这个交叉编译链工具适用于 Arm Cortex-M & Cortex-R processors (Cortex-M0/M0+/M3/M4/M7/M23/M33, Cortex-R4/R5/R7/R8/R52)系列平台。如何安装这个软件我就不赘述了，因为这个软件的安装没有什么需要注意的地方，在 Ubuntu 下只需要一句命令就可以了。

## 我们怎么开始

我们先联想一下用 keil_v5(MDK5) 编译 STM32 代码时的操作过程，如果没有用过这个软件，也可以联想 IAR 亦或是主机编译类软件的操作方式。 keil_v5 封装了编译链接的细节，使得我们觉得编译就是点一下按键的过程，根据我们在<a href="https://github.com/laneston/Note/blob/master/Hey_Makefile.md"> Hey Makefile! </a>的学习内容，我们认识到编译一个可执行的目标文件远比我们想象的要复杂得多。

正所谓万事开头难，中间难，结尾也难。那我们就从最简单的“点灯”程序开始吧，因为越简单的程序，给我们出错的余地就越少。我们先用 keil_v5 编译一个 LED 点灯程序，烧录到开发板中，以确保这个程序是可以正常工作的。代码的链接与详细说明请看这里：<a href = "https://github.com/laneston/STM32F4xx_LED-Makefile/tree/master/STM32F4xx_LED(keil_v5)">STM32F4xx_LED(keil_v5)</a>

代码可以通过验证，上面的工程已经能成功点亮一盏LED，并且以500ms的时间间隔闪烁。我们可以看到，这个工程中的代码根据功能不同来分类，分别存放在几个文件夹当中。

- STM32F4xx_DSP_StdPeriph_Lib_V1.8.0 [驱动内核与寄存机接口文件]
- Peripheral_BSP[外设驱动]
- User[代码入口]

主要代码分别放在以上3个文件夹当中，而其中一个文件夹 STM32F4xx_DSP_StdPeriph_Lib_V1.8.0 当中又有其他的文件夹。在当前的工程中，我们需要解决文件的依赖关系与文件路径寻找，如何让 Makefile 文件像 keil_v5 编译器那样，设置 include 路径后能自动查找 .c 文件所依赖的 .h 文件。

## 分析工程架构

```
STM32F4XX_LED(MAKEFILE)
    |__output
    |
    |__Peripheral_BSP
    |   |__stm32f4xx_led_bsp.c[Drive.c]
    |   |__stm32f4xx_led_bsp.h[Drive.h]
    |
    |__STM32F4xx_DSP_StdPeriph_Lib_V1.8.0
    |   |__CMSIS
    |   |   |__Device\ST\STM32F4xx
    |   |   |   |__Source\Templates
    |   |   |       |__arm
    |   |   |       |__gcc_ride7
    |   |   |       |__iar
    |   |   |       |__SW4STM32
    |   |   |       |__TSSKING
    |   |   |       |__TrueSTDIO
    |   |   |       |   |__startup_stm32f40_41xxx.s[Startup file]
    |   |   |       |
    |   |   |       |__system_stm32f4xx.c
    |   |   |
    |   |   |__Documentation
    |   |   |__DSP_Lib
    |   |   |__Include
    |   |   |__Lib
    |   |   |__RTOS_Lib
    |   |
    |   |__STM32F4x7_ETH_Driver
    |   |
    |   |__STM32F4xx_StdPeriph_Driver
    |       |__inc
    |       |__src
    |           |__misc.c
    |           |__stm32f4xx_gpio.c
    |           |__stm32f4xx_rcc.c
    |           |__stm32f4xx_tim.c
    |
    |
    |__User
    |   |__BSPConfig.h
    |   |__delay.c
    |   |__delay.h
    |   |__FreeRTOSConfig.h
    |   |__main.c
    |   |__main.h
    |   |__stm32f4xx_conf.h
    |   |__stm32f4xx_it.c
    |   |__stm32f4xx_it.h
    |   |__STM32F417IG_FLASH.ld
    |
    |__Makefile
```

根据以上的工程架构可以看出，这个代码框架其实很简单，与 <a href = "https://github.com/laneston/STM32F4xx_LED-Makefile/tree/master/STM32F4xx_LED(keil_v5)">STM32F4xx_LED(keil_v5)</a> 版本的代码没什么区别，但还是会有些许的不同。以下我就跟大家说明其中需要调整的部分。

框架中我特意标明了 .\STM32F4xx_DSP_StdPeriph_Lib_V1.8.0\CMSIS\Device\ST\STM32F4xx\Source\Templates\TrueSTUDIO\startup_stm32f40_41xxx.s 这一条路径，是因为我们不能选用 keil_v5 编译器用到的 STM32F4xx_DSP_StdPeriph_Lib_V1.8.0\CMSIS\Device\ST\STM32F4xx\Source\Templates\arm\startup_stm32f40_41xxx.s 这个文件。我们现在用到的 GCC 和 keil_v5 中所用的 ARMCC 编译工具是不同的，所以启动文件的编写有些许不同。在这里我想补充说明的是， keil 这个编译工具并不仅仅是把交叉编译工具封装成有 UI 界面的 IDE，其中做了很多的优化工作，而且 ARMCC 这个工具使用是需要付费的，所以这也是为什么 keil 卖那么贵的原因。最直接简单的对比就是，在当前的两个工程当中，用 keil_v5 编译的 bin 大小为 2KB，而用 arm-none-eabi- 编译的 bin 要去到 15 KB，在资源匮乏的 MUC 中，这个区别是致命的。啊，赤裸裸的金钱力量。当然，开源的力量也是强大的，后面我们稍作优化，也可以达到 3KB 的大小。

除此之外，我们还需要引入 STM32F417IG_FLASH.ld 这一个文件，这个文件我就不详细解释了，大家只需要知道这个文件是主要用来设置运行时用到的 SRAM 和 Flash 的参数。大家可以根据所用到的 MUC 对应修改参数。

还有，我们把 STM32F4xx_DSP_StdPeriph_Lib_V1.8.0\STM32F4xx_StdPeriph_Driver\src 里面的 .c 库文件缩减到需要用到的4个文件，那是因为如果把其他的文件添加进去，那么 Makefile 文件就因为要甄别需要用到的文件而变得十分复杂冗长，相比编写需要频繁修改的 Makefile 文件，增删几个文件就显得十分简单了。

# 编写Makefile

听我介绍完注意事项后，相信大家已经跃跃欲试编写一个 Makefile 文件了。以下我就直接给出完整的 Makefile 文件，内容并不复杂，只是有些事项需要解释。

```
BIN = ./output/STM32F4xx_LED.bin
ELF = ./output/STM32F4xx_LED.elf

FLASH_LD = ./User/STM32F417IG_FLASH.ld

SRC += ./User/*.c
SRC += ./Peripheral_BSP/*.c
SRC += ./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/STM32F4xx_StdPeriph_Driver/src/*.c
SRC += ./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Device/ST/STM32F4xx/Source/Templates/*.c

STARTUP = ./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Device/ST/STM32F4xx/Source/Templates/TrueSTUDIO/*.s

INC += -I./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/STM32F4xx_StdPeriph_Driver/inc
INC += -I./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Device/ST/STM32F4xx/Include
INC += -I./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Include
INC += -I./Peripheral_BSP
INC += -I./User

DEF += -DSTM32F40_41xxx 
DEF += -DUSE_STDPERIPH_DRIVER


CFLAGS += -mcpu=cortex-m4
CFLAGS += -mfloat-abi=hard
CFLAGS += -mthumb
CFLAGS += -mfpu=fpv4-sp-d16
CFLAGS += -Wall


LFLAGS += -mcpu=cortex-m4
LFLAGS += -mfloat-abi=hard
LFLAGS += -mthumb
LFLAGS += -mfpu=fpv4-sp-d16
LFLAGS += -Wl,--gc-sections

SRC_RAW = $(wildcard $(SRC))
OBJ = $(SRC_RAW:%.c=%.o)

STARTUP_RAW = $(wildcard $(STARTUP))
STARTUP_OBJ = $(STARTUP_RAW:%.s=%.o)

OUTPUT_OBJ = ./output/*.o

all:$(OBJ) $(STARTUP_OBJ) $(ELF) $(BIN)
	@echo $(ELF)
	@echo $(BIN)
	@echo

$(BIN):$(ELF)
	arm-none-eabi-objcopy -O binary -S $< $@
	@echo

$(ELF):
	arm-none-eabi-gcc -o $@ $(LFLAGS) $(OUTPUT_OBJ) -T$(FLASH_LD)
	@echo

%.o:%.s
	arm-none-eabi-gcc -c $(CFLAGS) $< -o $@
	-mv $@ ./output
	@echo

%.o:%.c
	arm-none-eabi-gcc -c $(CFLAGS) $(DEF) $(INC) $< -o $@
	-mv $@ ./output
	@echo

.PHONY: clean
clean :
	-rm ./output/*.o\
	    ./output/*.elf
```

Makefile 文件的逻辑我就不作解释了，因为写得十分简单，只要具备<a href="https://github.com/laneston/Note/blob/master/Hey_Makefile.md"> Hey Makefile! </a> 里面的知识，就足以应对了，以下我就跟大家说说其中用到命令和参数。

## 交叉编译链工具命令

我们这个项目只用到了两个命令，一个是把 .c .s 文件转换成 .o 文件。令一个是把执行文件 .elf 转换成单片机能够识别的 .bin 文件。

1. arm-none-eabi-gcc：将.c文件转化为.o的执行文件，用法与 GCC 工具一样。
2. arm-none-eabi-objcopy：生成可在 arm 平台上运行的bin文件，格式为：arm-linux-objcopy –O binary –S file.elf file.bin。

- -O bfdname 输出的格式
- -F bfdname 同时指明源文件,目的文件的格式
- -R sectionname 从输出文件中删除掉所有名为sectionname的段
- -S 不从源文件中复制重定位信息和符号信息到目标文件中
- -g 不从源文件中复制调试符号到目标文件中

## 编译参数

相信大家也看到了 CFLAGS 和 LFLAGS 两个变量，其实里面的参数与命令行中的 -o -c 这些指令参数没有什么差别。

1. -mcpu=cortex-m4： 这个不用我多说就能理解，因为我用的 MCU 是cortex-m4内核的，所以这里做了声明；
2. -mfloat-abi=hard： 这是浮点运算的指令参数，它的对立就是 soft ，因为MCU中有 FPU 单元，所以设置了硬件浮点；
3. -mthumb： 使用这个编译选项生成的目标文件是 thumb 指令集的；
4. -mfpu=fpv4-sp-d16： 浮点运算处理方式；
5. -Wl,--gc-sections： 在链接阶段，-Wl,–gc-sections 指示链接器去掉不用的代码（其中-wl, 表示后面的参数 -gc-sections 传递给链接器），这样就能减少最终的可执行程序的大小，且避免一些编译错误。

## 编译优化

到这里我们已经实现了交叉编译链下的STM32程序编译，但这时有朋友就不满意了，用 keil_v5 编译的执行文件只有 2KB ，而用 arm-none-eabi- 编译的居然有 15KB 那么大。别急，其实我们也能像编译器那样进行优化的。

一个东西为什么那么大？因为里面的东西多咯，如果里面有些东西我们不需要，那扔掉就可以减轻负担了。代码中也不是所有函数都会用到的，每个函数可以看作是一个section，GCC链接操作是以section作为最小的处理单元，只要一个section中的某个符号被引用，该section就会被加入到可执行程序中去。因此，GCC在编译时可以使用 -ffunction-sections和 -fdata-sections 将每个函数或符号创建为一个sections，其中每个sections名与function或data名保持一致。而在链接阶段， -Wl,–gc-sections 指示链接器去掉不用的section（其中-wl, 表示后面的参数 -gc-sections 传递给链接器），这样就能减少最终的可执行程序的大小了。所有最终 Makefile 就变成：

```
BIN = ./output/STM32F4xx_LED.bin
ELF = ./output/STM32F4xx_LED.elf

FLASH_LD = ./User/STM32F417IG_FLASH.ld

SRC += ./User/*.c
SRC += ./Peripheral_BSP/*.c
SRC += ./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/STM32F4xx_StdPeriph_Driver/src/*.c
SRC += ./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Device/ST/STM32F4xx/Source/Templates/*.c

STARTUP = ./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Device/ST/STM32F4xx/Source/Templates/TrueSTUDIO/*.s

INC += -I./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/STM32F4xx_StdPeriph_Driver/inc
INC += -I./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Device/ST/STM32F4xx/Include
INC += -I./STM32F4xx_DSP_StdPeriph_Lib_V1.8.0/CMSIS/Include
INC += -I./Peripheral_BSP
INC += -I./User

DEF += -DSTM32F40_41xxx 
DEF += -DUSE_STDPERIPH_DRIVER

CFLAGS += -mcpu=cortex-m4
CFLAGS += -mfloat-abi=hard
CFLAGS += -mthumb
CFLAGS += -mfpu=fpv4-sp-d16
CFLAGS += -Wall
CFLAGS += -Os
CFLAGS += -ffunction-sections
CFLAGS += -fdata-sections

LFLAGS += -mcpu=cortex-m4
LFLAGS += -mfloat-abi=hard
LFLAGS += -mthumb
LFLAGS += -mfpu=fpv4-sp-d16
LFLAGS += -Wl,--gc-sections

SRC_RAW = $(wildcard $(SRC))
OBJ = $(SRC_RAW:%.c=%.o)

STARTUP_RAW = $(wildcard $(STARTUP))
STARTUP_OBJ = $(STARTUP_RAW:%.s=%.o)

OUTPUT_OBJ = ./output/*.o

all:$(OBJ) $(STARTUP_OBJ) $(ELF) $(BIN)
	@echo $(ELF)
	@echo $(BIN)
	@echo

$(BIN):$(ELF)
	arm-none-eabi-objcopy -O binary -S $< $@
	@echo

$(ELF):
	arm-none-eabi-gcc -o $@ $(LFLAGS) $(OUTPUT_OBJ) -T$(FLASH_LD)
	@echo

%.o:%.s
	arm-none-eabi-gcc -c $(CFLAGS) $< -o $@
	-mv $@ ./output
	@echo

%.o:%.c
	arm-none-eabi-gcc -c $(CFLAGS) $(DEF) $(INC) $< -o $@
	-mv $@ ./output
	@echo

.PHONY: clean
clean :
	-rm ./output/*.o\
	    ./output/*.elf\
		./output/*.bin
```

补充说明一下，-Os 指令参数代表的是不同程度的编译优化：

- O0： 不做任何优化，这是默认的编译选项。
- O1： 优化会消耗少多的编译时间，它主要对代码的分支，常量以及表达式等进行优化。
- O2： 会尝试更多的寄存器级的优化以及指令级的优化，它会在编译期间占用更多的内存和编译时间。 
- O3： 在O2的基础上进行更多的优化
- Os： 是使用了所有-O2的优化选项，但又不缩减代码尺寸的方法。

经过这一番折腾，我们的 .bin 文件也得缩减到 3KB 大小。当然，其中一些操作也可以做的更巧妙些，arm-none-eabi- 工具链的功能也不止这些，但我们不在这里做讨论，因为这篇文章的目的就是让大家能快速上手交叉编译链下编写STM32的Makefile文件。