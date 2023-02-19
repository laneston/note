这篇文章主要用于讲述 arm for Linux 交叉编译链安装与使用，目的是使读者通过看完这篇博客后能够对交叉编译链有相关的了解，达到能够使用交叉编译链编译一个简单例程的效果，并在 S5P6818 平台上正常工作。

笔者之前在文章 **<a href = "https://github.com/laneston/Note/blob/master/Makefile_Note.md">交叉编译链下的Makefile</a>** 中讲述了 Cortex-M4 平台上的交叉编译方式，也大致说明了 arm-none-eabi-gcc 这款编译器的使用要点，现在我们继续从 Cortex-A53 平台的一系列编译工具出发，带大家深入了解交叉编译链是怎么在 x86-64 平台上编译出 arm 平台上的执行文件。

## 交叉编译器的命名规则及详细解释

一般来说，交叉编译工具链的命名规则为：arch-core-kernel-system-language。

其中：

- arch：体系架构，如ARM，MIPS，等，表示该编译器用于哪个目标平台；
- core：使用的是哪个CPU Core，如Cortex A8；或者是指定工具链的供应商。如果没有特殊指定，则留空不填。这一组命名比较灵活，在某些厂家提供的交叉编译链中，有以厂家名称命名的，也有以开发板命名的，或者直接是none或cross的；
- kernel： 所运行的OS，见过的有Linux，uclinux，bare（无OS）；
- system：交叉编译链所选择的库函数和目标映像的规范，如gnu，gnueabi等。其中gnu等价于glibc+oabi；gnueabi等价于glibc+eabi。若不指定，则也可以留空不填；
- language：编译语言，表示该编译器用于编译何种语言，最常见的就是gcc，g++；

### ABI与EABI

ABI：二进制应用程序接口（Application Binary Interface）。在计算机中，应用二进制接口描述了应用程序（或者其他类型）和操作系统之间或其他应用程序的低级接口；

EABI：即嵌入式ABI，应用于嵌入式系统的二进制应用程序接口（Embeded Application Binary Interface）。EABI指定了文件格式、数据类型、寄存器使用、堆积组织优化和在一个嵌入式软件中的参数的标准约定。开发者使用自己的汇编语言也可以使用 EABI 作为与兼容的编译器生成的汇编语言的接口。

两者主要区别是，ABI是计算机上的，EABI是嵌入式平台上（如ARM，MIPS等）。

### gnueabi与gnueabihf

- gcc-arm-linux-gnueabi    The GNU C compiler for armel architecture
- gcc-arm-linux-gnueabihf  The GNU C compiler for armhf architecture

可见这两个交叉编译器适用于 armel 和 armhf 两个不同的架构，armel 和 armhf 这两种架构在对待浮点运算采取了不同的策略（有fpu的arm才能支持这两种浮点运算策略）。

其实这两个交叉编译器只不过是 gcc 的选项 -mfloat-abi 的默认值不同。gcc 的选项 -mfloat-abi 有三种值soft、softfp、hard（其中后两者都要求arm里有fpu浮点运算单元，soft与后两者是兼容的，但softfp和hard两种模式互不兼容）：

soft：不用fpu进行浮点计算，即使有fpu浮点运算单元也不用，而是使用软件模式。
softfp：armel架构（对应的编译器为gcc-arm-linux-gnueabi）采用的默认值，用fpu计算，但是传参数用普通寄存器传，这样中断的时候，只需要保存普通寄存器，中断负荷小，但是参数需要转换成浮点的再计算。
hard：armhf架构（对应的编译器gcc-arm-linux-gnueabihf）采用的默认值，用fpu计算，传参数也用fpu中的浮点寄存器传，省去了转换，性能最好，但是中断负荷高。