本篇文章主要围绕项目 <a href = "https://github.com/laneston/STM32_RTOS_GUN">STM32_RTOS_GUN</a> 的链接脚本 STM32F417IG_FLASH.ld 进行分析，同时对编写链接脚本的方法进行相应的讲解，尽可能地做到通过阅读这篇文章后能够学会编写简单的链接脚本。

# 什么是链接脚本

我们在做 GCC 交叉编译时总会遇到一个 .ld 后缀的文件，这个就是我们常说的链接脚本。有朋友可能会疑惑这个脚本的到底有还是呢么用呢，我们在编写 GCC 命令时只需一句 gcc -o hello hello.c 就能同时实现而文件的编译和链接了，怎么在进行交叉编译时就需要这样也给文件呢？那是因为在编译主机运行的程序时，内存分配信息已经由编译器帮你写好了，而文件 STM32F417IG_FLASH.ld 恰恰是用来存放芯片内存信息的。

连接脚本的一个主要目的是描述输入文件中的节如何被映射到输出文件中，并控制输出文件的内存排布，几乎所有的连接脚本只做这两件事情。但是,在需要的时候，连接器脚本还可以指示连接器执行很多其他的操作。

连接器总是使用连接器脚本的，如果我们不提供，连接器会使用一个缺省的脚本，这个脚本是被编译进连接器可执行文件的。我们可以使用 '--verbose' 命令行选项来显示缺省的连接器脚本的内容。 某些命令行选项,比如 '-r' 或 '-N'，会影响缺省的连接脚本。

我们可以过使用 '-T' 命令行选项来提供你自己的连接脚本，就像我们 <a href = "https://github.com/laneston/STM32_RTOS_GUN">STM32_RTOS_GUN</a> 里 Makefile 文件写的那样。当这么做的时候, 连接脚本会替换缺省的连接脚本。

## 基本概念

总所周知，连接器把多个输入文件合并成单个输出文件。当中输出文件和输入文件都以一种叫做 “目标文件格式” 的可执行文件存在，譬如 .o 和 .elf 文件。每一个目标文件中都有一个节列表。我们把输入文件的节叫做输入节(input section)，相似的，输出文件中的一个节叫做输出节(output section)。目标文件的每个 section 至少包含两个信息: 名字和大小. 大部分 section 还包含与它相关联的一块数据, 称为 section contents(section内容)。 一个 section 可被标记为 “loadable(可加载的)” 或 “allocatable(可分配的)” 。这个概念笔者曾在编程笔记： <a href = "https://github.com/laneston/Note/blob/master/Makefile_Note.md">Makefile_Note</a>里 **编译优化** 一节中有提到过。

## VMA和LMA

每个“可加载的”或“可分配的”输出 section 通常包含两个地址: VMA(virtual memory address 虚拟内存地址或程序地址空间地址)和 LMA(load memory address 加载内存地址或进程地址空间地址)。通常 VMA 和 LMA 是相同的。

在目标文件中, loadable 或 allocatable 的输出 section 有两种地址: VMA(virtual Memory Address)和LMA(Load Memory Address)。 VMA是执行输出文件时section所在的地址，而LMA是加载输出文件时 section 所在的地址。一般而言，某 section 的VMA 与 LMA 相同。但在嵌入式系统中，经常存在加载地址和执行地址不同的情况: 比如将输出文件加载到开发板的 flash 中(由LMA指定)，而在运行时将位于 flash 中的输出文件复制到 RAM 中(由VMA指定)。我们把一些初始化的值先烧到 ROM 中，我们设定 LMA 为ROM 的地址，VMA 为 RAM 的地址。程序在内存中执行时，是根据 VMA 地址来执行的，也就是PC指针使用的是 VMA 地址空间。Loader下载程序时，Loader先将某些块程序拷贝到 ROM，但由于 ROM 只读，所以 ROM 的值不会变。接着比较 LMA 和 VMA，如果不相等，则把 LMA 处的内容拷贝到 VMA 所指向的位置，这样就把 ROM 处的值拷贝到了 RAM。

## 链接脚本

我们可以开始了解一个完整的脚本里应该包含什么样的内容，现在我们就来看一下 ST 的链接脚本文件 STM32F417IG_FLASH.ld

```
ENTRY(Reset_Handler)

_estack = 0x2001FFFF;        /* end of RAM */

_Min_Heap_Size = 0x200;      /* required amount of heap  */
_Min_Stack_Size = 0x400;     /* required amount of stack */

MEMORY
{
FLASH (rx)       : ORIGIN = 0x8000000, LENGTH = 1024K
RAM (xrw)        : ORIGIN = 0x20000000, LENGTH = 128K
CCMRAM (rw)      : ORIGIN = 0x10000000, LENGTH = 64K
}

SECTIONS
{
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)
    
    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >FLASH

  /* Constant data goes into FLASH */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  .ARM.extab   : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM : {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array     :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >FLASH

  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >FLASH

  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >FLASH

  _sidata = LOADADDR(.data);

  .data : 
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)

    . = ALIGN(4);
    _edata = .;
  } >RAM AT> FLASH

  _siccmram = LOADADDR(.ccmram);

  .ccmram :
  {
    . = ALIGN(4);
    _sccmram = .;
    *(.ccmram)
    *(.ccmram*)
    
    . = ALIGN(4);
    _eccmram = .;
  } >CCMRAM AT> FLASH

  . = ALIGN(4);
  .bss :
  {
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } >RAM

  ._user_heap_stack :
  {
    . = ALIGN(4);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(4);
  } >RAM  

  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}

```

这是 <a href = "https://github.com/laneston/STM32_RTOS_GUN">STM32_RTOS_GUN</a> 的链接脚本，项目基于 STM32F407 硬件平台进行开发，笔者把脚本中的注释去除掉了，让它看起来尽可能地简短，但我们看起来仍旧觉得凌乱不堪。其实链接脚本可以做到很简单，之所以举这个例子，是因为笔者经历过后思维比较清晰，而其中涉及的内容也较为普遍常用，没什么花里胡哨的技巧。

# 脚本分析与学习

在编译的过程中 .c 文件被编译为二进制执行文件 .o 文件，此时 .o 文件是未带有解析地址的，简而言之就是这个文件放在那个位置并未明确。而后工具链的链接器将所有C文件的 .o 文件链接将他们有序的排列到储存中，并将它们各个函数处的地址进行解析，使得其他不同地方的函数能够跳转到该函数的入口地址。由此一个有序排列的可被单片机执行的文件便生成了。至于其中各个.c 文件产生的功能在单片机储存中的排列顺序和地址位置，在最后我们链接器工作产生的.map文件中是有显示的，如下面从样例工程中.map文件中复制的片段：

## 简单脚本命令

我们就看脚本的第一行：

```
ENTRY(Reset_Handler)
```

从字面看，我们可以猜出这是一个 Reset 的中断句柄。指令被称为入口点 entry point，可以使用 ENTRY 链接脚本命令设置 entry point，参数是一个符号名。有几种方法可以设置 entry point,链接器会按照如下的顺序来尝试各种方法，只要任何一种方法成功就会停止：

1. ld 命令行的 -e 选项；
2. 连接脚本的ENTRY(SYMBOL)命令；
3. 如果定义了 start 符号，使用 start 符号值；
4. 如果存在 .text section，使用 .text section 的第一字节的位置值；
5. 使用值0。

```
_estack = 0x2001FFFF;
```

这段声明内存末尾地址。

```
_Min_Heap_Size = 0x200;
_Min_Stack_Size = 0x400;
```

这段定义了堆和栈的最小空间大小。如果定义的数值不符合内存的规格，在编译时会产生链接错误。

## MEMORY关键字

```
MEMORY
{
FLASH (rx)       : ORIGIN = 0x8000000, LENGTH = 1024K
RAM (xrw)        : ORIGIN = 0x20000000, LENGTH = 128K
CCMRAM (rw)      : ORIGIN = 0x10000000, LENGTH = 64K
}
```

这段定义了 FLASH RAM 和 CCMRAM 的起始地址和长度，(xrw)表明了权限，r是读、w是写、x是执行，这个和 Linux 中的 shell 命令一样。

连接器在缺省状态下被配置为允许分配所有可用的内存块，所以我们可以使用 ‘MEMORY’ 命令重新配置这个设置。‘MEMORY’ 命令描述目标平台上内存块的位置与长度。我们可以用它来描述哪些内存区域可以被连接器使用，哪些内存区域是要避免使用的，然后我们就可以把节(section)分配到特定的内存区域中。连接器会基于内存区域设置节的地址，对于太满的区域，会提示警告信息。连接器不会为了适应可用的区域而搅乱节。一个连接脚本最多可以包含一次 MEMORY 命令。但可以在命令中定义任意的内存块。

一旦你定义了一个内存区域，可以指示连接器把指定的输出段放入到这个内存区域中，这可以通过使用 ‘>REGION’ 输出段属性，这种操作可以在之后的内容中看到。

## SECTIONS

这个部分是 .ld 文件的核心部分，将会用较大篇幅去讲述，请各位看官耐心。

### SECTIONS 关键字

**SECTIONS 关键字** 是脚本文件中最重要的元素，不可缺省。它的作用就是用来描述输出文件的布局。

```
SECTIONS
{
    ...
    secname:{
        contents
    }
    ...
}
```

secname 和 contents 是必须的，其他都是可选的参数。SECTIONS 命令告诉 .ld 文件如何把输入文件的 sections 映射到输出文件的各个 section；如何将输入 section 合为输出 section；如何把输出 section 放入程序地址空间 (VMA) 和进程地址空间 (LMA) 。如果整个连接脚本内没有 SECTIONS 命令, 那么 .ld 将所有同名输入 section 合成为一个输出 section 内, 各输入 section 的顺序为它们被连接器发现的顺序。如果某输入 section 没有在 SECTIONS 命令中提到，那么该 section 将被直接拷贝成输出 section。

说到这里，很多朋友可能有点懵了，不过不要紧，毕竟掌握的信息还很少，尚不足以产生理解的质变，我们接着分析 SECTIONS 命令里面的内容。

### 输出 section 的描述

```
  .isr_vector:
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Startup code */
    . = ALIGN(4);
  } >FLASH
```

这几行脚本就是一个输出 section 的描述，我们可以看看其中的格式：

```
SECTION-NAME [ADDRESS] [(TYPE)] : [AT(LMA)]
{
  OUTPUT-SECTION-COMMAND
  OUTPUT-SECTION-COMMAND
  …
} [>REGION] [AT>LMA_REGION] [:PHDR HDR ...] [=FILLEXP]
```

我们看到上面的输出 section 命令与讲述的格式不太一样，少了许多东西，没错，中括号[]里的东西是可以省略的。但 .SECTION-NAME 左右的空白与冒号是必须的，所以上面的一段命令中 .SECTION-NAME 是 .isr_vector 。作为启动代码在 startup_stm32f40_41xxx.s 中定义： .section  .isr_vector,"a",%progbits

输出 section 名字 SECTION-NAME 必须符合输出文件格式要求，比如：.o 格式的文件只允许存在.text .data 和.bss 的 section 名。而有的格式只允许存在数字名字，那么此时应该用引号将所有名字内的数字组合在一起；另外，还有一些格式允许任何序列的字符存在于 section 名字内，此时如果名字内包含特殊字符(比如空格、逗号等)，那么需要用引号将其组合在一起。

输出 section 地址[ADDRESS]是一个表达式，它的值用于设置VMA。如果没有该选项且有 REGION 选项，那么连接器将根据 REGION 设置 VMA；如果也没有 REGION 选项，那么连接器将根据定位符号 ‘.’ 的值设置该 section 的 VMA，将定位符号的值调整到满足输出section 对齐要求后的值，这时输出 section 的对齐要求为：该输出 section 描述内用到的所有输入 section 的对齐要求中最严格的对齐要求。

. 小数点为地址定位器(counter)，表示当前的地址位置，并且可以赋值。

. = ALIGN(4); 是对齐格式，此处按4字节长度对齐。

>FLASH 意思是该 section 链接到 Flash 当中。

至于 KEEP(*(.isr_vector)) 这一句命令，我们需要分两部分来看：

- *(.isr_vector) 是通配符格式，表明的是所有输入文件的 .isr_vector section

- 在连接命令行内使用了选项 –gc-sections 后，连接器可能将某些它认为没用的 section 过滤掉，此时就有必要强制连接器保留一些特定的 section，可用 KEEP() 关键字达此目的。

整段的意思是：把所有输入文件(.o文件)的 .isr_vector 段命名为 .isr_vector section，并链接到 MEMORY 定义的 FLASH 中。

## 输出 section 的进阶描述

我们看以下这段命令：

```
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)
    
    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >FLASH
```

在进行分析之前先普及点基础知识：

1. .text   段是用来存放程序执行代码的区；
2. .data   段通常是用来存放程序中已初始化的全局变量的一块内存区域，属于静态内存分配。
3. .bss    段通常是指用来存放程序中未初始化的全局变量的一块内存区域，属于静态内存分配。
4. .rodata 段通常是指用来存放程序中常量的一块内存区域，属于静态内存分配。

\*(.text) 指示将工程中所有**输入文件** .o 的 代码段(.text) 链接到 MEMORY 定义的 FLASH 中； \*(.text*) 指示将工程中所有**目标文件**的 .text 段链接到 FLASH 中； *(.eh_frame) 指示输入文件所有的 .eh_frame 段链接到 FLASH 中。

通常编译器生成的代码会使用小的辅助函数来处理太大而无法内联的代码结构。例如处理器可能具有用于整数乘法的本机指令，但需要用于整数除法的辅助函数。gum_7t 段最适合用于 Thumb 模式的 ARMv7 核心粘合功能，而 gum_7 段则适用于 32 位 ARM 模式。

.eh_frame 用于处理异常，它产生描述如何展开堆栈的表。当使用 gcc 编译 C 程序并使用 readelf 检查 sections 信息时，可以看到其中的 .eh_frame sections 和 .eh_frame_hdr sections。

我们接着往下看：

```
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH
```

这段脚本的意思是：把输入文件(.o)和目标文件的常量区块(.rodata)链接到 Flash 中。

```
  .ARM.extab   : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM : {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH
```

.ARM.extab 包含异常信息的展开。

.ARM.exidx 是包含用于展开堆栈的信息的部分。如果你的 C 程序具有可打印出堆栈回溯的函数，则这些函数可能取决于存在的本节。exidx 是常规起始地址和 extab 索引的有序表。可以查验PC（程序计数器）并通过表格进行搜索以找到相应的extab条目。可以在编译器选项中用 -funwind-tables 或 -fexceptions 标志寻找。

```
  .preinit_array  :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >FLASH
```

这段脚本所描述的内容是将预初始化函数数组链接到 Flash 中。

PROVIDE_HIDDEN 关键字定义一个（目标文件内被引用但没定义）符号，相当于定义一个全局变量，其他C文件可以引用它。例如，传统的链接器定义了符号 “etext”。 但是，ANSI C 要求用户能够使用 “etext” 作为函数名称，而不会遇到错误。 仅当引用但未定义 PROVIDE 关键字时，才可以使用它来定义符号，就像 “etext”。 语法为 PROVIDE（symbol（symbol = expression）。而 PROVIDE_HIDDEN 与 PROVIDE 类似，作用相同。

```
  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >FLASH
  ```

这段脚本所描述的内容是将初始化函数数组链接到 Flash 中。

```
  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >FLASH
```

这段脚本所描述的是将析构或者说是删除的函数数组链接到 Flash 中。

看到这里，可能大家都还在云里雾里，不明白以上的脚本究竟有什么用，别急，下面我会继续给大家解析，但在解释以上脚本时我们先看看下面的符号：

```
/* These magic symbols are provided by the linker.  */
extern void (*__preinit_array_start []) (void) __attribute__((weak));
extern void (*__preinit_array_end []) (void) __attribute__((weak));
extern void (*__init_array_start []) (void) __attribute__((weak));
extern void (*__init_array_end []) (void) __attribute__((weak));
extern void (*__fini_array_start []) (void) __attribute__((weak));
extern void (*__fini_array_end []) (void) __attribute__((weak));
```

这些指针数组与 C/C++ 构造函数和析构函数的启动以及删除，以及在 main() 之前/之后调用的代码有关。名为.init，.ctors，.preinit_array 和 .init_array 的 section 与 C/C++ 对象的初始化有关，而 .fini，.fini_array 和 .dtors 的 section 用于删除或者说是析构。开头(__preinit_array_start)和结尾(__preinit_array_end)的符号用来定义与此类操作相关的代码 section 的开头和结尾，并且可以从运行时支持代码的其他部分中引用。

.preinit_array 和 .init_array 部分包含指向将在初始化时调用的函数的指针数组。.fini_array 是在销毁时调用的函数数组，而开始和结束标签用于遍历这些列表。

```
  _sidata = LOADADDR(.data);

  .data : 
  {
    . = ALIGN(4);
    _sdata = .;        /* create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data*)          /* .data* sections */

    . = ALIGN(4);
    _edata = .;        /* define a global symbol at data end */
  } >RAM AT> FLASH
```

.data section 包含所有初始化的全局变量和静态变量，启动代码将从 .data section 的 LMA 复制到 .data section 的 VMA。 为了进一步说明这一点，.data section 的静态变量和全局变量需要存储在两个不同的位置：

1. VMA（虚拟内存地址）：编译后代码的变量的运行地址。 这在RAM中，以 > RAM 表示。
2. LMA（装载存储器地址）：初始化数据存储的地址。 这在 Flash 中以 AT 表示，如果不添加 AT 这个命令，编译地址将会连续，中间数据会以0来填充，生成的 .bin 文件将会变得极大。

```
 _siccmram = LOADADDR(.ccmram);

  .ccmram :
  {
    . = ALIGN(4);
    _sccmram = .;       /* create a global symbol at ccmram start */
    *(.ccmram)
    *(.ccmram*)
    
    . = ALIGN(4);
    _eccmram = .;       /* create a global symbol at ccmram end */
  } >CCMRAM AT> FLASH
```

这段脚本用于在链接器中指明，代码段将从 .ccmram section 的 LMA 复制到 .ccmram section 的 VMA。但若想使用 MCU 的 CCM RAM 仍需做额外的修改处理，我们现在看的 .ld 文件是 ST 默认的，只能使用 128KB 的 SRAM，没法使用 64KB CCM SRAM。修改内容我已在 <a href = "https://github.com/laneston/STM32_RTOS_GUN">STM32_RTOS_GUN</a> 的 README 中说明。

```
  . = ALIGN(4);
  .bss :
  {
    /* This is used by the startup in order to initialize the .bss secion */
    _sbss = .;         /* define a global symbol at bss start */
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;         /* define a global symbol at bss end */
    __bss_end__ = _ebss;
  } >RAM
  ```

这段脚本用来将程序中未初始化的全局变量链接到 RAM 中。

```
  ._user_heap_stack :
  {
    . = ALIGN(4);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(4);
  } >RAM  
```

这部分用于检查是否还有足够的RAM。

```
  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }
```

这是特殊输出 section 名称 / DISCARD / ，可用于丢弃输入节。任何分配给名为“ / DISCARD /”的输出节的输入节不包含在输出文件中。简而言之，这段脚本是用户删除标准库编译的中间信息。

到这里为止我们基本上将 STM32F4 的 LD 文件解析完毕了。但我想很多朋友还是不太明白，因为其中很多内容是来自 <a href = "https://sourceware.org/binutils/docs/ld/"> GNU Binutils </a>，这篇文章只是基于这个手册的部分内容进行简单说明，至于更加详细的操作事项，需要阅读这个手册。