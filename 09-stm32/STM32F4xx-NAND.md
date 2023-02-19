**目录**

<a href="#STM32F4xx Part">STM32F4xx Part</a><br>
- <a href="#8-bit NAND Flash">8-bit NAND Flash</a>
- <a href="#NAND address mapping">NAND address mapping</a>
- <a href="#NAND Flash operations">NAND Flash operations</a>
- <a href="#Timing diagrams for NAND">Timing diagrams for NAND</a>
- <a href="#Error Correction Code">Error Correction Code (ECC)</a>
- <a href="#Timing diagrams for NAND">Timing diagrams for NAND</a>
- <a href="#NAND Flash prewait functionality">NAND Flash prewait functionality</a>
- <a href="#NAND Flash Card control registers">NAND Flash Card control registers</a>
  1. <a href="#FSMC_PCR">PC Card/NAND Flash control registers 2..4 (FSMC_PCR2..4)</a>
  2. <a href="#FSMC_SR">FIFO status and interrupt register 2..4 (FSMC_SR2..4)</a>
  3. <a href="#FSMC_PMEM">Common memory space timing register 2..4 (FSMC_PMEM2..4)</a>
  4. <a href="#FSMC_PATT">Attribute memory space timing registers 2..4 (FSMC_PATT2..4)</a>
  5. <a href="#FSMC_PIO4">I/O space timing register 4 (FSMC_PIO4)</a>
  6. <a href="#FSMC_ECCR">ECC result registers 2/3 (FSMC_ECCR2/3)</a>

<a href="#MX30LF1G18AC Part">MX30LF1G18AC Part</a><br>
- <a href="#Timing Configuration">Timing Configuration</a>
- <a href="#Address Assignment">Address Assignment</a>
- <a href="#Precautions">Precautions</a>

 <h1 id="STM32F4xx Part"> STM32F4xx Part</h1>

这一部分内容是关于STM32的灵活静态存储控制器（FSMC）在NAND Flash上的应用方式与注意事项。

 <h3 id="8-bit NAND Flash">8-bit NAND Flash</h3>
 
| FSMC signal name | I/O | Function                                             |
| :--------------: | --- | ---------------------------------------------------- |
|      A[17]       | O   | NAND Flash address latch enable (ALE) signal         |
|      A[16]       | O   | NAND Flash command latch enable (CLE) signal         |
|      D[7:0]      | I/O | 8-bit multiplexed, bidirectional address/data bus    |
|      NCE[x]      | O   | Chip select, x = 2, 3                                |
|    NOE(= NRE)    | O   | Output enable (memory signal name: read enable, NRE) |
|       NWE        | O   | Write enable                                         |
|  NWAIT/INT[3:2]  | I   | NAND Flash ready/busy input signal to the FSMC       |

**Programmable NAND/PC Card access parameters**

| Parameter             | Function                                                       | Access mode | Unit                   | Min. | Max. |
| :-------------------- | :------------------------------------------------------------- | :---------- | :--------------------- | :--- | :--- |
| Memory setup time     | 在命令生效之前设置地址的时钟周期数（HCLK）                     | Read/Write  | AHB clock cycle (HCLK) | 1    | 255  |
| Memory wait           | 命令生效的最短持续时钟周期                                     | Read/Write  | AHB clock cycle (HCLK) | 2    | 256  |
| Memory hold           | 在命令失效之后，保持地址以及写入访问的数据的时钟周期数（HCLK） | Read/Write  | AHB clock cycle (HCLK) | 1    | 254  |
| Memory databus high-Z | 开始一个写入访问后，数据总线保持high-Z状态的时钟周期数（HCLK） | Write       | AHB clock cycle (HCLK) | 0    | 255  |

<h3 id="NAND address mapping">NAND address mapping</h3>

| Start address | End address | FSMC Bank           | Memory space | Timing register   |
| :-----------: | :---------: | :------------------ | :----------- | :---------------- |
|  0x8800 0000  | 0x8BFF FFFF | Bank 3 - NAND Flash | Attribute    | FSMC_PATT3 (0x8C) |
|  0x8000 0000  | 0x83FF FFFF | Bank 3 - NAND Flash | Common       | FSMC_PMEM3 (0x88) |
|  0x7800 0000  | 0x7BFF FFFF | Bank 2 - NAND Flash | Attribute    | FSMC_PATT2 (0x6C) |
|  0x7000 0000  | 0x73FF FFFF | Bank 2 - NAND Flash | Common       | FSMC_PMEM2 (0x68) |

对于 NAND Flash memory，通用存储空间和属性存储空间分为三个部分，位于较低的256 KB中：

- 数据部分：first 64 Kbytes in the common/attribute memory space
- 命令部分：second 64 Kbytes in the common / attribute memory space
- 地址部分：next 128 Kbytes in the common / attribute memory space

| Section name    | HADDR[17:16] |   Address range   |
| :-------------- | :----------: | :---------------: |
| Address section |      1X      | 0x020000-0x03FFFF |
| Command section |      01      | 0x010000-0x01FFFF |
| Data section    |      00      | 0x000000-0x0FFFF  |

应用程序使用这3部分访问 NAND Flash memory。

**发送命令到 NAND Flash memory：** 要执行发送命令操作，只需要把命令发送至Command section部分任意一个地址上。

**指定 NAND Flash 地址：** 要执行发送地址操作，只需要把地址发送至Address section部分任意一个地址上。地址可以为4或5个字节长度，这取决于实际内存大小，要指定完整地址，需要对地址部分进行多次连续写入。

**读或写数据：** 要执行读或写数据的操作，只需要读或写Data section部分任意一个地址。

由于NAND闪存会自动递增地址，因此无需增加数据段的地址来访问连续的内存位置，即如果访问连续地址的数据段，只需要连续读或写Data section部分的地址即可。

<h3 id="NAND Flash operations">NAND Flash operations</h3>

NAND闪存设备的命令锁存使能（CLE）和地址锁存使能（ALE）信号由 FMC 控制器的一些地址信号驱动。这意味着要向NAND闪存发送命令或地址，CPU必须对其内存空间中的某个地址执行写入操作。

NAND闪存设备的典型页面读取操作如下：

**Step 1：** 

通过配置寄存器 FSMC_PCRx 和 FSMC_PMEMx，一些设备需要配置FSMC_PATTx，来使能和配置相应的 memory bank。

**Step 2：** 

CPU在公共内存空间执行字节写入，数据字节等于一个闪存命令字节（例如，对于Samsung NAND闪存设备，为0x00）。NAND闪存的CLE输入在写入选通（NWE上的低脉冲）期间处于活动状态，因此写入的字节被解释为NAND闪存的命令。一旦命令被NAND闪存设备锁定，就不需要为以下页面读取操作写入该命令。

**Step 3：** 

CPU可以通过写入4个字节的数据发送起始地址（STARTAD）来实行读取操作，对于容量较小的设备，3个或更少的字节足矣。字节写入公共内存空间（common memory ）或属性内存空间（attribute space）。

STARTAD[7:0], STARTAD[16:9],STARTAD[24:17] and finally STARTAD[25](for 64 Mb x 8 bit NAND Flash memories)

NAND Flash 设备的ALE输入在写入（NWE低脉冲）期间处于活动状态，因此写入的字节被解释为读取操作的起始地址。使用属性内存空间可以使用FMC的不同定时配置，该配置可用于实现某些NAND闪存所需的预等待功能。

**Step 4：** 

开始一个新的链接到相同或不同的 memory bank 之前，控制器等待 NAND Flash memory 变成就绪状态（R/NB 信号为高电平）。在等待过程中，控制器保持NCE信号为活跃状态（低电平）

**Step 5：** 

然后，CPU可以在公共内存空间中执行字节读取操作，逐字节读取NAND闪存页（数据字段+备用字段）。

**Step 6：**

下一个NAND闪存页可以在没有任何CPU命令或地址写入操作的情况下以三种不同的方式读取：

1. 只需执行步骤5中描述的操作；
2. 可以通过在步骤3重新启动操作来访问新的随机地址；
3. 通过在步骤2重新启动，可以向NAND闪存设备发送新命令。

<h3 id="Error Correction Code">Error Correction Code (ECC)</h3>

FSMC PC卡控制器包括两个纠错码计算硬件块，每个存储库一个。它们用于减少系统软件处理纠错码时主机CPU的工作量。这两个寄存器相同，分别与bank 2和bank 3相关联。因此，没有硬件ECC计算可用于连接到bank 4的存储器。

FSMC中实现的纠错码（ECC）算法可以对从 NAND Flash 读写的256、512、1024、2048、4096或8192个字节执行1位纠错和2位错误检测。它基于汉明编码算法，包括行和列奇偶校验的计算。

每次 NAND Flash 组激活时，ECC模块监测 NAND Flash 数据总线和读/写信号（NCE和NWE）。

功能操作包括：

- 当对bank 2和bank 3进行NAND闪存访问时，D[15:0]总线上的数据被锁定并用于ECC计算。
- 在任何地址访问NAND闪存时，ECC逻辑处于空闲状态，不执行任何操作。因此，定义命令或地址的写操作不考虑在ECC计算中。

一旦CPU从NAND闪存读取/写入所需的字节数，必须读取FSMC_ECCR2/3寄存器以检索计算值。一旦读取，应通过将ECCEN位重置为零来清除它们。要计算新的数据块，ECCEN位必须在FSMC_PCR2/3寄存器中设置为1。

执行ECC计算：

1.	在FSMC_PCR2/3寄存器中启用ECCEN位。
2.	将数据写入NAND闪存页。写入NAND页时，ECC块计算ECC值。
3.	读取FSMC_ECCR2/3寄存器中可用的ECC值，并将其存储在变量中。
4.	清除ECCEN位，然后在FSMC_PCR2/3寄存器中启用它，然后从NAND页读回写入的数据。在读取NAND页时，ECC块计算ECC值。
5.	ECC寄存器3中可用的新ECC/R2值。
6.	如果两个ECC值相同，则不需要进行校正，否则会出现ECC错误，并且软件校正例程返回有关错误是否可以校正的信息。

 <h3 id="Timing diagrams for NAND">Timing diagrams for NAND</h3>

每个PC卡/CompactFlash和NAND闪存库通过一组寄存器进行管理：

- 控制寄存器：FSMC_PCRx
- 中断状态寄存器：FSMC_SRx
- 纠错码（EEC）寄存器：FSMC_ECCRx
- 公共存储空间定时寄存器：FSMC_PMEMx
- 属性存储空间定时寄存器：FSMC_PATTx
- I/O空间定时寄存器：FSMC_PIOx

每个定时配置寄存器包含三个参数，用于定义NAND Flash 访问的三个阶段的HCLK循环数，外加一个参数，用于定义在写入访问时，开始驱动数据总线的定时。

公共内存访问，属性存储空间访和I/O存储空间访问（仅适用于PC卡）内存空间访问时序是相似的。

下图显示了参数的定义：

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xx_NAND/NAND%20controller%20timing%20for%20common%20memory%20access.jpg"></div>

1.	在写入访问期间，NOE保持高（非活动）。在读取访问期间，NWE保持高（非活动）。
2.	对于写访问，保持阶段延迟为（MEMHOLD）x HCLK周期，而对于读访问，保持阶段延迟为（MEMHOLD+2）x HCLK周期。

 <h3 id="NAND Flash prewait functionality">NAND Flash prewait functionality</h3>

一些 NAND Flash 设备要求在写入最后一部分地址后，控制器等待 R/NB 信号电平变低[busy signal]。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xx_NAND/Access%20to%20non%20%E2%80%98CE%20don%E2%80%99t%20care%E2%80%99%20NAND-Flash.jpg"></div>

1. CPU在地址 0x7001 0000 写字节 0x00;
2. CPU在地址 0x7002 0000 写字节 A7-A0;
3. CPU在地址 0x7002 0000 写字节 A15-A8;
4. CPU在地址 0x7002 0000 写字节 A23-A16;
5. CPU在地址 0x7802 0000 写字节 A25-A24;

FSMC 使用 FSMC_PATT2 寄存器的时序定义执行一次写访问，其中ATTHOLD≥7（保证（7+1）×HCLK = 112 ns>tWB max）。这保证了NCE保持在低电平，直到R/NB再次拉低拉高（为了使NCE不受BUSY信号影响）。

当需要此功能时，可以通过设置 MEMHOLD 值来满足 tWB 定时要求。但是，CPU 对 NAND 的读取访问的保持延迟为（MEMHOLD+2）x HCLK cycles，CPU 对 NAND 的写入访问的保持延迟为（MEMHOLD）x HCLK周期。

为了克服这种定时限制，可以使用属性内存空间，方法是用满足 tWB 定时的 ATTHOLD 值对其定时寄存器进行编程，并将MEMHOLD值保持在最小值。然后，CPU必须对所有 NAND 的读写访问使用公共内存空间，除非将最后地址字节写入NAND Flash设备时，CPU 必须将其写入属性内存空间。

<h3 id="NAND Flash Card control registers">NAND Flash Card control registers</h3>

NAND控制寄存器必须通过字（32位）访问。

<h4 id="FSMC_PCR">PC Card/NAND Flash control registers 2..4 (FSMC_PCR2..4)</h4>

Address offset: 0xA0000000 + 0x40 + 0x20 * (x – 1), x = 2..4

Reset value: 0x0000 0018

**Bit 31:20** 保留，必须保持在重置值

**Bit 19:17 ECCPS[2:0]:** ECC 页大小

定义扩展ECC的页大小：

- 000: 256 bytes
- 001: 512 bytes
- 010: 1024 bytes
- 011: 2048 bytes
- 100: 4096 bytes
- 101: 8192 bytes

**Bit 16:13 TAR[2:0]:** ALE 到 RE(read enable)的延迟

跟据AHB时钟(HCLK)循环数量设置从ALE变低到RE变低的时间。

计算公式： t_ar = (TAR + SET + 2) × THCLK (这里的THCLK是HCLK的周期时长)

- 0000: 1 HCLK cycle (default)
- 1111: 16 HCLK cycles

Note：根据寻址空间，SET是MEMSET或ATTSET。

**Bit 12:9 TCLR[2:0]:** CLE 到 RE(read enable)的延迟

跟据AHB时钟(HCLK)循环数量设置从CLE变低到RE变低的时间。

计算公式：t_clr = (TCLR + SET + 2) × THCLK (这里的THCLK是HCLK的周期时长)

- 0000: 1 HCLK cycle (default)
- 1111: 16 HCLK cycles

Note：根据寻址空间，SET是MEMSET或ATTSET。

**Bit 8:7** 保留，必须保持在重置值

**Bit 6 ECCEN:** ECC计算逻辑使能位

- 0: ECC logic is disabled and reset (default after reset),
- 1: ECC logic is enabled.

**Bit 5:4 PWID[1:0]:** 数据总线宽度

定义外部内存设备宽度。

- 00: 8 bits
- 01: 16 bits (default after reset). This value is mandatory for PC Cards.
- 10: reserved, do not use
- 11: reserved, do not use

**Bit 3 PTYP:** Memory 类型

定义连接到相应memory bank的设备类型：

- 0: PC Card, CompactFlash, CF+ or PCMCIA
- 1: NAND Flash (default after reset)

**Bit 2 PBKEN:** PC Card/NAND Flash memory bank使能位。

使能memory bank。访问使能的memory bank会导致AHB总线上出现错误。
- 0：对应的memory bank被禁用（复位后默认）
- 1：启用相应的memory bank

**Bit 1 PWAITEN:**  等待功能使能位

使能PC Card/NAND Flash memory bank的等待功能：
- 0: disabled
- 1: enabled

对于PC卡，当启用等待功能时，MEMWAITx/ATTWAITx/IOWAITx位必须编程为如下值：
xxWAITx≥4+max_wait_assertion_time/HCLK

其中max_wait_assertion_time是nOE/nWE或nIORD/nIOWR低时NWAIT进入低位所用的最长时间。

**Bit 0** 保留，必须保持在重置值

--------------------------

<h4 id="FSMC_SR">FIFO status and interrupt register 2..4 (FSMC_SR2..4)</h4>

Address offset: 0xA000 0000 + 0x44 + 0x20 * (x-1), x = 2..4

Reset value: 0x0000 0040

该寄存器包含有关FIFO状态和中断的信息。FSMC有一个FIFO，在写入存储器时用来存储来自AHB的多达16个字的数据。这用于快速写入AHB，并将其释放到FSMC以外的外围设备，同时FSMC正在将其FIFO排入内存。出于ECC目的，该寄存器有一个位指示FIFO的状态。ECC是在数据写入内存时计算的，因此为了读取正确的ECC，软件必须等到FIFO为空。

**Bit 31:7** 保留，必须保持在重置值

**Bit 6 FEMPT:** FIFO状态为空标志位

提供FIFO状态的只读位

- 0: FIFO 不为空
- 1: FIFO 为空

**Bit 5 IFEN:** 下降沿中断信号检测启用位

- 0: Interrupt falling edge detection request disabled
- 1: Interrupt falling edge detection request enabled

**Bit 4 ILEN:** 高级中断检测启用位

- 0: Interrupt high-level detection request disabled
- 1: Interrupt high-level detection request enabled

**Bit 3 IREN:** 上升沿中断信号检测启用位

- 0: Interrupt rising edge detection request disabled
- 1: Interrupt rising edge detection request enabled

**Bit 2 IFS:** 下降沿中断标志位

这个标志位硬件置1，需要被软件复位。

- 0: No interrupt falling edge occurred
- 1: Interrupt falling edge occurred

**Bit 1 ILS:** 高级中断标志位

这个标志位硬件置1，需要被软件复位。

- 0: No Interrupt high-level occurred
- 1: Interrupt high-level occurred

**Bit 0 IRS:** 上升沿中断标志位

这个标志位硬件置1，需要被软件复位。

- 0: No interrupt rising edge occurred
- 1: Interrupt rising edge occurred

--------------------------

<h4 id="FSMC_PMEM">Common memory space timing register 2..4 (FSMC_PMEM2..4)</h4>

Address offset: Address: 0xA000 0000 + 0x48 + 0x20 * (x – 1), x = 2..4

Reset value: 0xFCFC FCFC

每个FSMC_PMEMx（x=2..4）读/写寄存器包含PC卡或NAND闪存组x的定时信息，用于访问16位PC卡/CompactFlash的公共存储空间，或访问NAND Flash进行命令、地址写入访问和数据读/写访问。

**Bit 31:24 MEMHIZx[7:0]** 通用存储器x的数据总线 HiZ time

对socket x上的公共内存空间进行NAND flash 写入访问开始后，定义数据总线保存在HiZ中的HCLK时钟周期数。

仅对写入事务有效：

- 0000 0000: 1 HCLK cycle
- 1111 1110: 255 HCLK cycles
- 1111 1111: Reserved

**Bit 23:16 MEMHOLDx[7:0]** 通用存储器x的 hold time

对于对公共内存空间的NAND读取访问，在命令被解除(NWE, NOE)之后地址被保持，这些位定义了（HCLK+2）时钟周期数。

对于对公共内存空间的NAND写入访问，在命令被解除(NWE, NOE)之后数据被保持，这些位定义了 HCLK 时钟周期数。

- 0000 0000: Reserved
- 0000 0001: 1 HCLK cycle for write accesses, 3 HCLK cycles for read accesses
- 1111 1110: 254 HCLK cycle for write accesses, 256 HCLK cycles for read accesses
- 1111 1111: Reserved

**Bit 15:8 MEMWAITx[7:0]** 通用存储器x的 wait time

对socket x上的公共内存空间进行NAND读写访问，定义最小HCLK（+1）时钟周期数去使用命令(NWE,NOE)。如果等待信号（NWAIT）在HCLK编程值结束时激活（电平变低），命令激活的持续时间被延长。

- 0000 0000: Reserved
- 0000 0001: 2 HCLK cycles (+ wait cycle introduced by deasserting NWAIT)
- 1111 1110: 255 HCLK cycles (+ wait cycle introduced by deasserting NWAIT)
- 1111 1111: Reserved.

**Bit 15:8 MEMSETx[7:0]** 通用存储器x的 setup time

在命令激活（NWE，NOE）之前，定义HCLK（）时钟周期数去设置地址，对socket x上公共内存空间的NAND读写访问：

- 0000 0000: 1 HCLK cycle
- 1111 1110: 255 HCLK cycles
- 1111 1111: Reserved

--------------------------

<h4 id="FSMC_PATT">Attribute memory space timing registers 2..4 (FSMC_PATT2..4)</h4>

Address offset: 0xA000 0000 + 0x4C + 0x20 * (x – 1), x = 2..4

Reset value: 0xFCFC FCFC

每个FSMC_PATTx（x=2..4）读/写寄存器包含 PC卡/CompactFlash 或 NAND Flash bank x 的定时信息。如果时间必须与以前的访问不同，用于对属性内存空间进行8位访问，以对NAND进行最后一次地址写入访问。

**Bit 31:24 MEMHIZx[7:0]** 属性存储器x的数据总线 HiZ time

对socket x上的公共内存空间进行NAND flash 写入访问开始后，定义数据总线保存在HiZ中的HCLK时钟周期数。

仅对写入事务有效：

- 0000 0000: 1 HCLK cycle
- 1111 1110: 255 HCLK cycles
- 1111 1111: Reserved

**Bit 23:16 MEMHOLDx[7:0]** 属性存储器x的 hold time

对于对公共内存空间的NAND读取访问，在命令被解除(NWE, NOE)之后地址被保持，这些位定义了（HCLK+2）时钟周期数。

对于对公共内存空间的NAND写入访问，在命令被解除(NWE, NOE)之后数据被保持，这些位定义了 HCLK 时钟周期数。

- 0000 0000: Reserved
- 0000 0001: 1 HCLK cycle for write accesses, 3 HCLK cycles for read accesses
- 1111 1110: 254 HCLK cycle for write accesses, 256 HCLK cycles for read accesses
- 1111 1111: Reserved

**Bit 15:8 MEMWAITx[7:0]** 属性存储器x的 wait time

对socket x上的公共内存空间进行NAND读写访问，定义最小HCLK（+1）时钟周期数去使用命令(NWE,NOE)。如果等待信号（NWAIT）在HCLK编程值结束时激活（电平变低），命令激活的持续时间被延长。

- 0000 0000: Reserved
- 0000 0001: 2 HCLK cycles (+ wait cycle introduced by deasserting NWAIT)
- 1111 1110: 255 HCLK cycles (+ wait cycle introduced by deasserting NWAIT)
- 1111 1111: Reserved.

**Bit 15:8 MEMSETx[7:0]** 属性存储器x的 setup time

在命令激活（NWE，NOE）之前，定义HCLK（）时钟周期数去设置地址，对socket x上公共内存空间的NAND读写访问：

- 0000 0000: 1 HCLK cycle
- 1111 1110: 255 HCLK cycles
- 1111 1111: Reserved

--------------------------

<h4 id="FSMC_PIO4">I/O space timing register 4 (FSMC_PIO4)</h4>

Address offset: 0xA000 0000 + 0xB0

Reset value: 0xFCFCFCFC

FSMC_PIO4读/写寄存器包含用于访问16位PC卡/CompactFlash的I/O空间的计时信息。

[与NAND Flash 无关，略过]

--------------------------

<h4 id="FSMC_ECCR">ECC result registers 2/3 (FSMC_ECCR2/3)</h4>

Address offset: 0xA000 0000 + 0x54 + 0x20 * (x – 1), x = 2 or 3

Reset value: 0x0000 0000

这些寄存器包含FSMC控制器的ECC计算模块（每个NAND Flash memory bank一个模块）计算的当前纠错代码值。当CPU以正确的地址从 NAND Flash 存储器页读取数据时，ECC计算模块自动处理从 NAND Flash 读写的数据。在X字节读取结束时，CPU必须从FSMC_ECCx寄存器中读取计算出的ECC值，然后验证这些计算出的奇偶校验数据是否与备用区中记录的奇偶校验值相同，以确定某个页是否有效，并在适用的情况下对其进行纠正。FSMC_ECCRx寄存器应在读取后通过将ECCEN位设置为零来清除。要计算新的数据块，ECCEN位必须设置为1。

**Bit 31:0 ECCx[31:0]:** ECC result

此字段提供ECC计算逻辑计算的值。下表描述了这些位字段的内容：

| ECCPS[2:0] | Page size in bytes | ECC bits  |
| :--------- | :----------------- | :-------- |
| 000        | 256                | ECC[21:0] |
| 001        | 512                | ECC[23:0] |
| 010        | 1024               | ECC[25:0] |
| 001        | 2048               | ECC[27:0] |
| 100        | 4096               | ECC[29:0] |
| 101        | 8192               | ECC[31:0] |


<h1 id="MX30LF1G18AC Part">MX30LF1G18AC Part</h1>

MX30LF1G18AC 由64页（2048+64）字节组成，采用两个NAND字符串结构，每个字符串中有32个串行连接单元。每一页都有额外的64字节用于ECC和其他用途。该设备具有2112字节的片上缓冲区，用于数据加载和访问，每个2K字节的缓冲页面有两个区域，一个是2048字节的主区域，另一个是64字节的备用区域。

<h3 id="Timing Configuration"> Timing Configuration </h3>

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xx_NAND/AC%20Waveforms%20for%20Command-Address-Data%20Latch%20Timing.jpg"></div>

上图是项目所用到的NAND Flash MX30LF1G18AC的 Address Input/Command Input/Data Input时序图。图中标注意义如下：

| symbol | detail           | value |
| :----- | :--------------- | :---: |
| tCS    | CE# setup time   | >15ns |
| tCLS   | CLE setup time   | >10ns |
| tALS   | ALE setup time   | >10ns |
| tCH    | #CE hold time    | >5ns  |
| tCLH   | CLE hold time    | >5ns  |
| tWP    | write pulse time | >10ns |
| tDS    | DATA setup time  | >7ns  |
| tDH    | DATA hold time   | >5ns  |

**tCS/tCLS/tALS = (MEMxSET+1) + (MEMxWAIT+1)**

**tCH/tCLH = MEMxHOLD**

**tWP = MEMxWAIT + 1**

**tCS/tCLS/tALS - tWP = MEMxHIZ**

<h3 id="Address Assignment"> Address Assignment </h3>

地址分配有四个地址周期：

| Addresses                  |  IO7  |  IO6  |  IO5  |  IO4  |  IO3  |  IO2  |  IO1  |  IO0  |
| :------------------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Column address - 1st cycle |  A7   |  A6   |  A5   |  A4   |  A3   |  A2   |  A1   |  A0   |
| Column address - 2nd cycle |   L   |   L   |   L   |   L   |  A11  |  A10  |  A9   |  A8   |
| Row address - 3rd cycle    |  A19  |  A18  |  A17  |  A16  |  A15  |  A14  |  A13  |  A12  |
| Row address - 4th cycle    |  A27  |  A26  |  A25  |  A24  |  A23  |  A22  |  A21  |  A20  |

MX30xx系列设备是顺序存取存储器，利用多路复用x8或x16输入/输出总线上的命令/地址/数据信号的多路输入。此接口减少了管脚数，并使迁移到其他密度而不改变封装外形成为可能。

1. 地址输入总线操作是为地址输入选择存储器地址；
2. 令输入总线操作用于向存储器发出命令；
3. 数据输入总线用于向存储设备输入数据。

<h3 id="Precautions"> Precautions </h3>

当芯片输入电压达到上电水平（Vth=Vcc min.）后，将触发内部上电复位序列。在内部上电复位期间，不接受任何外部命令。有两种方法可以识别内部通电复位序列的终止。

- R/B# pin
- Wait 1 ms

在通电和断电过程中，建议保持WP#=低，以保护内部数据。

WP#信号保持低位，存储器将不接受程序/擦除操作。在通电/断电过程中，建议将WP引脚保持在低位。