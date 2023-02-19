**目录**

<a href="#Ethernet main features">Ethernet main features</a>
- <a href="#MAC core features">MAC core features</a>
- <a href="#DMA features">DMA features</a>
- <a href="#PTP features">PTP features</a>

<a href="#SMI MII and RMII">SMI, MII and RMII</a>
- <a href="#Station management interface">Station management interface: SMI</a>
- <a href="#Media-independent interface">Media-independent interface: MII</a>
- <a href="#Ethernet main features">Reduced media-independent interface: RMII</a>

<a href="#MAC frame transmission">MAC frame transmission</a>
- <a href="#Automatic CRC and pad genratione">Automatic CRC and pad genratione</a>
- <a href="#Transmit protocol"> Transmit protocol </a>
- <a href="#Transmit scheduler"> Transmit scheduler </a>
- <a href="#Transmit flow control"> Transmit flow control </a>
- <a href="#Transmit operation—Two packets in the buffer">Transmit operation—Two packets in the buffer</a>
- <a href="#Retransmission during collision"> Retransmission during collision </a>
- <a href="#Transmit FIFO flush operation"> Transmit FIFO flush operation </a>
- <a href="#Transmit status word"> Transmit status word </a>
- <a href="#Transmit checksum offload"> Transmit checksum offload </a>
- <a href="#MII/RMII transmit bit order"> MII/RMII transmit bit order </a>

<a href="#MAC frame transmission">MAC frame transmission</a>
- <a href="#Receive protocol">Receive protocol</a>
- <a href="#Receive CRC">Receive CRC: automatic CRC and pad stripping</a>
- <a href="#Receive checksum offload">Receive checksum offload</a>
- <a href="#Receive flow control">Receive flow control</a>
- <a href="#Receive operation multiframe handling">Receive operation multiframe handling</a>
- <a href="#Error handling">Error handling</a>
- <a href="#Receive status word">Receive status word</a>
- <a href="#Frame length interface">Frame length interface</a>
- <a href="#MII/RMII receive bit order">MII/RMII receive bit order</a>

<a href="#Ethernet functional description">Ethernet functional description</a>
- <a href="#Initialization of a transfer using DMA">Initialization of a transfer using DMA</a>
- <a href="#Host bus burst access">Host bus burst access</a>
- <a href="#Host data buffer alignment">Host data buffer alignment</a>
- <a href="#Buffer size calculations">Buffer size calculations</a>
- <a href="#DMA arbiter">DMA arbiter</a>
- <a href="#Error response to DMA">Error response to DMA</a>
- <a href="#Tx DMA configuration">Tx DMA configuration</a>
  - <a href="#default mode">TxDMA operation: default (non-OSF) mode</a>
  - <a href="#OSF mode">TxDMA operation: OSF mode</a>


# 序

本篇文章是翻译ST手册 RM0090 以太网硬件部分，本着学习的目的，笔者会在翻译的过程添加自己的见解。由于篇幅较长，翻译需要分多次完成，主要会围绕当前开源项目的内容进行学习与翻译，其余非主要内容会在文章收尾之后补上。


<h1 id="Ethernet main features"> Ethernet main features </h1>

以太网外备使STM32F4xx能够根据IEEE 802.3-2002标准通过以太网发送和接收数据。它设提供了一个可配置的、灵活的外围设备，以满足各种应用和客户的需求。它支持到外部物理层（PHY）的两个行业标准接口：IEEE 802.3规范中定义的默认媒体独立接口（MII）和精简媒体独立接口（RMII）。它可以用于许多应用，如交换机、网络接口卡等。

<h3 id="MAC core features"> MAC core features </h3>

1. 有外部PHY接口并支持10/100mbit/s的数据传输速率。

2. 符合IEEE 802.3标准的MII接口，用于与外部快速以太网PHY通信。

3. 支持全双工和半双工操作：

- 支持用于半双工操作的CSMA/CD协议；

- 支持全双工操作的IEEE 802.3x流量控制；

- 在全双工操作中，可选地将接收到的暂停控制帧转发给用户应用程序；

- 背压支持半双工操作；

- 在全双工操作中，流量控制输入解除时，自动发送零量程暂停帧。

4. 在发送中插入前导码和帧起始数据（SFD），在接收路径中删除。

5. 在每帧的基础上实行自动CRC校验和生成PAD。（以太网帧长度最小64字节，长度不够的需要加PAD。）

6. 接收帧上的自动PAD/CRC剥离选项。

7. 可编程的帧长度，支持最大为16kb的标准帧。

8. 可编程帧间隙（40-96bit 时长）。

9. 支持多种灵活的地址筛选模式：

- 最多4个48位完美（DA）地址过滤器，每个字节都有掩码；

- 最多3个48位SA地址比较检查，每个字节都有掩码；

- 用于多播和单播（DA）地址的64位哈希过滤器（可选）；

- 通过所有多播寻址帧的选项；

- 支持混杂模式，无需任何过滤即可通过所有帧进行网络监控；

- 使用状态报告传递所有传入数据包（根据筛选器）。

10. 为发送和接收数据包返回单独的状态位（32bit）。

11. 支持接收帧的IEEE 802.1Q VLAN标记检测。

12. 独立的传输和接收以及应用控制接口。

13. 支持带有RMON/MIB计数器（RFC2819/RFC2665）的强制网络统计信息。

14. 用于PHY设备配置和管理的MDIO接口。

15. LAN唤醒帧和AMD Magic Packet帧检测。

16. 以太网帧封装的已接收IPv4和TCP数据包的校验和卸载接收功能。

17. 增强的接收功能，用于检查IPv4头校验和以及封装在IPv4或IPv6数据报中的TCP、UDP或ICMP的校验和。

18. 支持IEEE1588-2008中描述的以太网帧时间戳。在每帧的发送或接收状态下给出64位时间戳。

19. 两组FIFO：一个具有可编程阈值能力的2-KB传输FIFO，一个具有可配置阈值的2-KB接收FIFO（默认为64字节）

20. 在EOF传输后插入接收FIFO的接收状态向量允许在接收FIFO中存储多个帧，而不需要另一个FIFO来存储这些帧的接收状态。

21. 选项在接收时过滤所有错误帧，而不在存储和转发模式下将其转发到应用程序。

22. 选择转发尺寸不足的好帧。

23. 通过为接收队列中丢失或损坏的帧（由于溢出）生成脉冲来支持统计。

24. 支持向MAC核传输的存储转发机制。

25. 基于接收队列填充（阈值可配置）水平，自动生成暂停帧控制或返回到MAC核心的压力信号。

26. 处理碰撞帧的自动重新传输以进行传输。

27. 在后期碰撞、过度碰撞、过度延迟和欠载情况下丢弃帧。

28. 刷新Tx 队列的软件控制。

29. 存储和转发模式下，在传输的帧中计算并插入IPv4头校验和和和TCP、UDP或ICMP的校验和。

30. 支持MII上的内部环回以进行调试。

<h3 id="DMA features"> DMA features </h3>

1. 支持AHB从接口中的所有AHB突发类型。

2. 软件可在AHB主界面中选择AHB突发类型（固定或不定突发）。

3. 从AHB主端口选择地址对齐脉冲的选项。

4. 使用帧分隔符优化面向分组的DMA传输。

5. 字节对齐寻址，支持数据缓冲区。

6. 双缓冲区（环）或链表（链式）描述符链式。

7. 描述符架构，允许大数据块传输，CPU干预最少。

8. 每个描述符最多可以传输8 KB的数据。

9. 全面的正常运行和错误传输状态报告。

10. 用于发送和接收DMA引擎的单个可编程突发大小，以实现最佳主机总线利用率。

11. 不同操作条件下的可编程中断选项。

12. 每帧发送/接收完全中断控制。

13. 接收和发送引擎之间的循环或固定优先级仲裁。

14. 启动/停止模式。

15. 当前Tx/Rx缓冲指针作为状态寄存器。

16. 当前Tx/Rx描述符指针作为状态寄存器。

<h3 id="PTP features"> PTP features </h3>

1. 接收和发送帧时间戳。

2. 粗、精校正方法。

3. 当系统时间大于目标时间时触发中断。

4. 每秒脉冲输出（产品替代功能输出）。 

<h1 id="SMI MII and RMII"> SMI, MII and RMII </h1>

以太网外围设备由带有专用DMA控制器的MAC 802.3（媒体访问控制）组成。它通过一个选择位（参考SYSCFG_PMC寄存器）支持默认的媒体独立接口（MII）和精简的媒体独立接口（RMII）。

DMA控制器通过AHB主从接口与核心和存储器接口。AHB主接口控制数据传输，而AHB从接口访问控制和状态寄存器（CSR）空间。

发送FIFO（Tx FIFO）缓冲在MAC核发送前由DMA从系统存储器读取数据。类似地，接收FIFO（Rx FIFO）存储接收的以太网帧，直到它们被DMA传输到系统存储器。

以太网外围设备还包括SMI以与外部PHY通信。一组配置寄存器允许用户为MAC和DMA控制器选择所需的模式和特性。

注：使用以太网时，AHB时钟频率必须至少为25 MHz。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/ETH%20block%20diagram.jpg"></div>

<h3 id="Station management interface"> Station management interface: SMI </h3>

<h3 id="Media-independent interface"> Media-independent interface: MII </h3>

<h3 id="Reduced media-independent interface"> Reduced media-independent interface: RMII </h3>

精简的媒体独立接口（RMII）规范以10/100mbit/s的速度减少了微控制器以太网外围设备和外部以太网之间的管脚数。

根据IEEE 802.3u标准，MII包含16个数据和控制管脚。RMII规范专门用于将管脚数减少到7个管脚（管脚数减少62.5%）。

RMII在MAC和PHY之间实例化。这有助于将MAC的MII转换为RMII。RMII块具有以下特征：

- 支持10 Mbit/s和100 Mbit/s的工作速率；

- 时钟基准必须加倍至50 MHz；

- 相同的时钟参考必须从外部来源到MAC和外部以太网PHY；

- 它提供独立的2位宽（dibit）的传输和接收数据路径。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Reduced%20media-independent%20interface%20signals.png"></div>

<h3 id="RMII clock sources"> RMII clock sources </h3>

从外部50MHz时钟对PHY进行时钟，或者使用带有嵌入式PLL的PHY来生成50MHz频率。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/RMII%20clock%20sources.png"></div>

<h3 id="MII/RMII selection"> MII/RMII selection </h3>

模式MII或RMII是使用SYSCFG_PMC寄存器中的配置位23 MII_RMII_SEL来选择的。当以太网控制器处于重置状态或启用时钟之前，应用程序必须设置MII/RMII模式。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Clock%20scheme.png"></div>

为了节省一个引脚，两个输入时钟信号RMII_REF_CK和MII_RX_CLK被多路复用在同一个GPIO管脚上。

<h1 id="MAC 802.3"> MAC 802.3 </h1>

IEEE 802.3局域网（LANs）国际标准采用CSMA/CD（带冲突检测的载波感知多址）作为接入方式。以太网外设由独立接口（MII）的MAC 802.3（媒体访问控制）控制器和专用DMA控制器组成。

MAC区块为以下系列系统实现LAN-CSMA/CD子层：基带和宽带系统的10mbit/s和100mbit/s数据速率。支持半双工和全双工操作模式。碰撞检测访问方法仅适用于半双工操作模式。支持MAC控制帧子层。

MAC子层执行与数据链路控制过程相关联的以下功能：

1. 数据封装（发送和接收）

- 帧（帧边界定界、帧同步）

- 寻址（处理源地址和目标地址）

- 错误检测

2. 媒体访问管理

- 中等配置（避免碰撞）

- 争用解决（冲突处理）

基本上，MAC子层有两种工作模式：

1. 半双工模式：工作站使用CSMA/CD算法争夺物理介质的使用。

2. 全双工模式：当满足以下所有条件时，无需争用资源（不需要CSMA/CD算法）的同时传输和接收：

- 支持同时传输和接收的物理介质能力

- 正好有2个站点连接到LAN

- 两个站点均配置为全双工操作

<h3 id="MAC 802.3 frame format"> MAC 802.3 frame format </h3>

MAC块实现IEEE 802.3-2002标准指定的MAC子层和可选MAC控制子层（10/100mbit/s）。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/MAC%20frame%20format.jpg"></div>

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Tagged%20MAC%20frame%20format.png"></div>

为使用CSMA/CD MAC的数据通信系统指定了两种帧格式：

1. 基本MAC帧格式

2. 标记的MAC帧格式（基本MAC帧格式的扩展）

以上两张图片描述了框架结构（未标记和标记），包括以下字段：

1. 前导码3：用于同步的7字节字段（PLS电路）十六进制值：55-55-55-55-55-55-55-55-55位模式：01010101 01010101 01010101 01010101 01010101 01010101（从右到左位传输）。

2. 开始帧分隔符（SFD）：用于指示帧开始的1字节字段。十六进制值：D5位模式：11010101（从右到左位传输）。

3. 目的地和源地址字段：6字节字段，表示目的地和源站地址，如下所示）：

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Address%20field%20format.png"></div>

- 每个地址的长度为48位。

- 目标地址字段中的第一个LSB位（I/G）用于指示个人地址（I/G=0）或组地址（I/G=1）。一个组地址能识别无、1个、多个或所有连接到局域网的站点。在源地址中，第一位被保留并重置为0。

- 第二位（U/L）区分本地（U/L=1）或全局（U/L=0）受管地址。对于广播地址，该位也是1。

- 每个地址字段的每个字节必须首先传输最低有效位。

地址指定基于以下类型：

1. 单个地址：这是与网络上特定站点相关联的物理地址。

2. 组地址。与给定网络上的一个或多个站点关联的多目标地址。多播地址有两种：

- 多播组地址：与一组逻辑相关站点关联的地址。

- 广播地址：一个可分辨的预定义多播地址（目标地址字段中的所有1），它始终表示给定LAN上的所有站点。

3. QTag前缀：在源地址字段和MAC客户端长度/类型字段之间插入的4字节字段。此字段是基本帧（未标记）的扩展，用于获取标记的MAC帧。未标记的MAC帧不包括此字段。标记扩展如下：

- 与类型解释（大于0x0600）一致的2字节常量长度/类型字段值，等于802.1Q标记协议类型（0x8100十六进制）的值。此常量字段用于区分标记的和未标记的MAC帧。

- 包含标签控制信息字段的2字节字段，细分如下：3位用户优先级、规范格式指示符（CFI）位和12位VLAN标识符。标记的MAC帧的长度由QTag前缀扩展了4个字节。

4. MAC客户端长度/类型：2字节字段，含义不同（互斥），具体取决于其值：

- 如果该值小于或等于maxValidFrame（0d1500），则此字段指示802.3帧的后续数据字段中包含的MAC客户端数据字节数（长度解释）。

- 如果该值大于或等于MinTypeValue（0d1536 decimal，0x0600），则此字段指示与以太网帧相关的MAC客户端协议（类型解释）的性质。

无论对长度/类型字段的解释如何，如果数据字段的长度小于协议正常运行所需的最小值，则在数据字段之后但在FCS（帧检查序列）字段之前添加一个PAD字段。长度/类型字段首先用高阶字节发送和接收。

对于maxValidLength和minTypeValue（不包括边界）之间的长度/类型字段值，不指定MAC子层的行为：MAC子层可以传递它们，也可以不传递它们。

5. 数据和PAD字段：n字节数据字段。提供了完全的数据透明性，这意味着任何字节值的任意序列都可能出现在数据字段中。PAD的大小（如果有的话）由数据字段的大小决定。数据和PAD字段的最大和最小长度为：

- 最大长度=1500字节

- 未标记MAC帧的最小长度=46字节

- 标记MAC帧的最小长度=42字节

当数据字段长度小于所需的最小值时，将添加PAD字段以匹配最小长度（标记帧为42字节，未标记帧为46字节）。

6. 帧检查序列：包含循环冗余检查（CRC）值的4字节字段。CRC计算基于以下字段：源地址、目标地址、QTag前缀、长度/类型、LLC数据和PAD（即除前导码、SFD之外的所有字段）。生成多项式如下：

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Formula_20200702A.jpg"></div>

帧的CRC值计算如下：

1. 帧的前2位被补。

2. 帧的n位是次数(n–1)的多项式M(x)的系数。目标地址的第一位对应于x^(n-1)项，数据字段的最后一位对应于x^0项。

3. M(x)与x^32相乘，再除以G(x)，产生一个小于31的余数。

4. R(x)的系数被视为32位序列。

5. 位序列被补充，结果是CRC。

6. CRC值的32位放入帧检查序列中。x^32项是第一个传输，x^0项是最后一个传输。

MAC帧的每个字节，除FCS字段外，首先传输低阶位。无效的MAC帧由以下条件之一定义：

1. 帧长度与长度/类型字段指定的预期值不一致。如果长度/类型字段包含类型值，则假定帧长度与此字段一致（没有无效帧）。

2. 帧长度不是整数字节数（额外位）。

3. 在传入帧上计算的CRC值与包含的FCS不匹配。

<h1 id="MAC frame transmission"> MAC frame transmission </h1>

DMA控制传输路径的所有事务。从系统存储器读取的以太网帧被DMA推入FIFO。然后将这些帧弹出并传输到MAC核心。当帧结束被传输时，传输的状态从MAC核取出并传输回DMA。传输FIFO的深度为2Kbyte。FIFO填充级别指示给DMA，以便它可以使用AHB接口从系统内存启动所需的突发数据提取。来自AHB主接口的数据被推入FIFO。

当检测到SOF时，MAC接收数据并开始向MII传输。应用程序启动传输后，将帧数据传输到MII所需的时间是可变的，这取决于IFG延迟、发送前导码/SFD的时间以及半双工模式的任何退避延迟等延迟因素。EOF传输到MAC核后，核心完成正常传输，然后将传输状态返回给DMA。如果在传输过程中发生正常的冲突（半双工模式），MAC核将使传输状态有效，然后接受并丢弃所有进一步的数据，直到接收到下一个SOF。在观察到来自MAC的重试请求（处于状态）时，应该从SOF重新传输相同的帧。如果在传输期间没有连续地提供数据，MAC发出下溢状态。在帧的正常传输过程中，如果MAC接收到SOF而没有得到前一帧的EOF，则SOF被忽略，新帧被视为前一帧的继续。

向MAC核弹出数据有两种操作模式：

1. 在阈值模式下，只要FIFO中的字节数超过配置的阈值级别（或在超过阈值之前写入帧结束时），数据就可以弹出并转发到MAC核心。使用ETH-DMABMR的TTC位配置阈值大小。

2. 在存储和转发模式下，只有在FIFO中存储完整帧后，帧才会朝MAC核弹出。如果Tx FIFO的大小小于要传输的以太网帧，则当Tx FIFO几乎满时，该帧会向MAC核弹出。

应用程序可以通过设置FTF（ETH-DMAOMR register[20]）bit来刷新所有内容的传输FIFO。该位是自清除的，并将FIFO指针初始化为默认状态。如果在帧传输到MAC核期间设置了FTF位，则传输将停止，因为FIFO被认为是空的。因此，在MAC发射机处发生下溢事件，并且相应的状态字被转发到DMA。

<h3 id="Automatic CRC and pad genratione"> Automatic CRC and pad generation </h3>

当从应用程序接收到的字节数低于60（DA+SA+LT+Data）时，在发送帧中追加零，使数据长度正好为46字节，以满足IEEE 802.3的最小数据字段要求。MAC可以编程为不附加任何填充。计算帧检查序列（FCS）字段的循环冗余校验（CRC），并将其附加到正在发送的数据中。当MAC被编程为不将CRC值附加到以太网帧的末尾时，计算出的CRC不会被发送。此规则的一个例外是，当MAC被编程为对小于60字节的帧（DA+SA+LT+Data）附加PAD时，CRC将被附加到填充帧的末尾。

CRC生成器计算以太网帧的FCS字段的32位CRC。编码由以下多项式定义。

 <div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Formula_20200702A.jpg"></div>

<h3 id="Transmit protocol"> Transmit protocol </h3>

MAC控制以太网帧传输的操作。它执行以下功能以满足IEEE 802.3/802.3z规范。

1. 生成前导码和SFD。

2. 以半双工模式生成阻塞模式。

3. 控制Jabber超时。

4. 控制半双工模式的流量（背压式）。

5. 生成传输帧状态。

6. 包含符合IEEE1588的时间戳快照逻辑。

当请求新的帧传输时，MAC发送前导码和SFD，然后是数据。前导码被定义为0b10101010模式的7字节，SFD被定义为0b10101011模式的1字节。碰撞窗口定义为1时隙时间（10/100mbit/s以太网为512位时间）。干扰模式生成仅适用于半双工模式，而不适用于全双工模式。

在MII模式下，如果在从帧开始到CRC字段结束的任何时候发生冲突，MAC在MII上发送一个0x555555的32位阻塞模式，以通知所有其他站点发生了冲突。如果在前导码传输阶段发现冲突，MAC完成前导码和SFD的传输，然后发送干扰模式。

如果必须传输超过2048字节（默认值），则保持jabber定时器以切断以太网帧的传输。MAC在半双工模式下使用延迟机制进行流量控制（背压）。当应用程序请求停止接收帧时，MAC只要感测到帧的接收，就发送32字节的干扰模式，前提是发射流控制被启用。

这会导致碰撞，远程工作站会后退。应用程序通过在ETH-MACFCR寄存器中设置BPA位（位0）来请求流控制。如果应用程序请求传输帧，则即使在启动背压时也会对其进行调度和传输。注意，如果背压保持激活很长一段时间（并且发生超过16个连续的碰撞事件），则远程站会因过度碰撞而中止其传输。如果为传输帧启用IEEE1588时间戳，则当SFD被放到传输MII总线上时，此块将获取系统时间的快照。

<h3 id="Transmit scheduler"> Transmit scheduler </h3>

MAC负责调度MII上的帧传输。它保持两个传输帧之间的帧间隔，并遵循半双工模式下的截断二元指数退避算法。MAC在满足IFG和退避延迟后启用传输。它保持任意两个发送帧之间配置的帧间间隔（ETH-MACCR寄存器中的IFG位）的空闲周期。如果要发送的帧早于配置的IFG时间到达，则MII在开始对其发送之前等待来自MAC的使能信号。一旦MII的载波信号变为非活动状态，MAC就会启动IFG计数器。在编程的IFG值结束时，MAC启用全双工模式的传输。

在半双工模式下，当IFG被配置为96位次时，MAC遵循IEEE 802.3规范第4.2.3.2.1节中规定的遵从规则。如果在IFG间隔的前三分之二（所有IFG值为64位倍）期间检测到载波，MAC重置其IFG计数器。如果在IFG间隔的最后三分之一期间检测到载波，MAC继续IFG计数并在IFG间隔之后启用发射机。MAC在半双工模式下运行时实现了截短的二进制指数退避算法。

<h3 id="Transmit flow control"> Transmit flow control </h3>

当发送流控制使能位（ETH-MACFCR中的TFE位）被设置时，MAC生成暂停帧，并在必要时以全双工模式发送它们。暂停帧被附加计算出的CRC，并被发送。暂停帧生成可以通过两种方式启动。

当应用程序在ETH-MACFCR寄存器中设置FCB位或当接收FIFO已满（包缓冲区）时，发送暂停帧。

如果应用程序已通过在ETH-MACFCR中设置FCB位来请求流控制，则MAC生成并发送单个暂停帧。生成帧中的暂停时间值包含ETH_MACFCR中的编程暂停时间值。要在先前发送的暂停帧中指定的时间之前延长暂停或结束暂停，应用程序必须在将暂停时间值（ETH_MACFCR寄存器中的PT）编程为适当的值之后请求另一个暂停帧传输。

如果应用程序在接收FIFO满时请求流控制，MAC生成并发送暂停帧。生成帧中的暂停时间的值是ETH-MACFCR中的编程暂停时间值。如果在该暂停时间耗尽之前，接收FIFO在可配置的时隙次数（ETH_MACFCR中的PLT比特）处保持满，则发送第二暂停帧。只要接收的FIFO保持满，则重复该过程。如果在采样时间之前不再满足该条件，则MAC发送具有零暂停时间的暂停帧，以指示远程端接收缓冲器准备好接收新的数据帧。

<h3 id="Single-packet transmit operation"> Single-packet transmit operation </h3>

发送操作的一般事件顺序如下：

1. 如果系统有要传输的数据，DMA控制器通过AHB主接口从内存中获取这些数据，并开始将它们转发到FIFO。它继续接收数据，直到传输完帧。

2. 当超过阈值水平或接收到FIFO中的完整数据包时，帧数据被弹出并驱动到MAC核。DMA继续从FIFO传输数据，直到完整的数据包传输到MAC。当帧完成时，DMA控制器被来自MAC的状态通知。

<h3 id="Transmit operation—Two packets in the buffer"> Transmit operation—Two packets in the buffer </h3>

1. 因为DMA必须在将描述符释放到主机之前更新描述符状态，所以在传输FIFO中最多可以有两个帧。只有在设置了OSF（对第二帧操作）位的情况下，DMA才会获取第二帧并将其放入FIFO。如果未设置此位，则只有在MAC完全处理完该帧并且DMA释放了描述符之后，才从内存中获取下一帧。

2. 如果设置了OSF位，DMA在完成第一帧到FIFO的传输后立即开始获取第二帧。它不会等待状态更新。同时，当第一帧被发送时，第二帧被接收到FIFO中。一旦第一帧被传输并且从MAC接收到状态，它就被推送到DMA。如果DMA已经完成将第二个包发送到FIFO，则第二个传输必须等待第一个包的状态，然后才能继续到下一帧。

<h3 id="Retransmission during collision"> Retransmission during collision </h3>

当帧被传送到MAC时，在半双工模式下MAC线路接口上可能发生冲突事件。然后，MAC将通过在接收到帧结束之前，给出状态来指示重试尝试。然后重新传输被启用，帧从FIFO中再次弹出。在向MAC核弹出超过96个字节后，FIFO控制器释放出空间，并使DMA可以将更多数据推入。这意味着超过此阈值或当MAC核指示延迟碰撞事件时，无法重新传输。

<h3 id="Transmit FIFO flush operation"> Transmit FIFO flush operation </h3>

MAC通过使用操作模式寄存器中的位20来控制软件刷新发送FIFO。刷新操作是立即的，即使Tx FIFO正在向MAC核传输帧，Tx FIFO和相应的指针也被清除到初始状态。这导致MAC发送器中发生下溢事件，并且帧传输被中止。这种帧的状态用下溢和帧刷新事件（TDES0位13和1）标记。在刷新操作期间，没有数据从应用程序（DMA）进入FIFO。传输传输状态字被传输到应用程序以获得刷新的帧数（包括部分帧数）。完全刷新的帧设置了帧刷新状态位（TDES0 13）。当应用程序（DMA）已接受刷新帧的所有状态字时，刷新操作完成。然后清除传输FIFO刷新控制寄存器位。此时，来自应用程序（DMA）的新帧被接受。所有在刷新操作后显示以供传输的数据都将被丢弃，除非它们以SOF标记开头。

<h3 id="Transmit status word"> Transmit status word </h3>

在以太网帧传输到MAC核心的最后，在核心完成帧的传输之后，向应用程序给出传输状态。传输状态的详细描述与TDES0中的位[23:0]相同。如果启用IEEE1588时间戳，则返回特定帧的64位时间戳以及传输状态。

<h3 id="Transmit checksum offload"> Transmit checksum offload </h3>

TCP和UDP等通信协议实现校验和字段，这有助于确定通过网络传输的数据的完整性。由于以太网最广泛的用途是封装TCP和UDP over IP数据报，因此以太网控制器具有传输校验和卸载功能，支持在传输路径中进行校验和计算和插入，并在接收路径中进行错误检测。本节说明传输帧的校验和卸载功能的操作。

**TCP、UDP或ICMP的校验和是在一个完整的帧上计算的，然后插入到其相应的头字段中。由于这一要求，仅当发送FIFO被配置为存储和转发模式时（即，当TSF位设置在ETH-ETH-DMAOMR寄存器中时），此功能才被启用。如果核心配置为阈值（穿透）模式，则绕过传输校验和卸载。**

**在将帧传输到MAC核心发射机之前，必须确保传输FIFO足够深，能够存储完整的帧。如果FIFO深度小于输入以太网帧大小，则绕过有效负载（TCP/UDP/ICMP）校验和插入功能，仅修改帧的IPv4头校验和，即使在存储和转发模式下也是如此。**

**传输校验和卸载支持两种校验和计算插入。这个可以通过设置CIC位（位28:27 in TDES1，在TDES1:Transmit descriptor Word1中描述）来控制每个帧的校验和。**

*有关IPv4、TCP、UDP、ICMP、IPv6和ICMPv6数据包头规范，请参见IETF规范RFC 791、RFC 793、RFC 768、RFC 792、RFC 2460和RFC 4443。*

<h3 id="MII/RMII transmit bit order"> MII/RMII transmit bit order </h3>

每个来自MII的半字节在RMII上发送出去，一次两位，其传输顺序如下图所示。低阶位（D1和D0）首先传输，然后传输高阶位（D2和D3）。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Transmission%20bit%20order.jpg"></div>

下图是RMII的帧传输。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Frame%20transmission%20in%20MMI%20and%20RMII%20modes.jpg"></div>

<h1 id="MAC frame reception">MAC frame reception</h1>

MAC接收到的帧会被推入Rx FIFO。一旦这个FIFO超过配置的接收阈值（ETH_DMAOMR寄存器中的RTC），DMA就可以向AHB接口发起预配置的突发传输。

在默认的 **Cut-through** 模式下，当64个字节（在ETH DMAOMR寄存器中配置了RTC位）或一个完整的数据包被接收到FIFO中时，数据被弹出到内存中，并通知DMA Rx FIFO 可用性。一旦DMA启动，向AHB接口传输数据，数据将从FIFO持续传输，直到完整的数据包传输完毕。EOF帧传输完成后，状态字弹出并发送给DMA控制器。

在Rx FIFO 的 **Store-and-forward** 模式（由ETH DMAOMR寄存器中的RSF位配置）中，帧在完全写入接收FIFO之后才被读取。在这种模式下，所有错误帧都会被丢弃（如果内核配置为这样做），这样只读取有效帧并将其转发给应用程序。在 **Cut-through** 模式下，一些错误帧不会被丢弃，因为错误状态是在帧的末尾接收到的，此时该帧的开始已经被FIFO读取。

当MAC在MII上检测到SFD时，将启动接收操作。在继续处理帧之前，核心将去除前导码和SFD。检查头字段是否过滤，FCS字段用于验证帧的CRC。如果地址筛选器失败，则将帧丢弃在核心中。

<h3 id="Receive protocol"> Receive protocol </h3>

[待翻译]

<h3 id="Receive CRC"> Receive CRC: automatic CRC and pad stripping </h3>

[待翻译]

<h3 id="Receive checksum offload"> Receive checksum offload </h3>

[待翻译]

<h3 id="Receive flow control"> Receive flow control </h3>

[待翻译]

<h3 id="Receive operation multiframe handling"> Receive operation multiframe handling </h3>

[待翻译]

<h3 id="Error handling"> Error handling </h3>

[待翻译]

<h3 id="Receive status word"> Receive status word </h3>

[待翻译]

<h3 id="Frame length interface"> Frame length interface </h3>

[待翻译]

<h3 id="MII/RMII receive bit order"> MII/RMII receive bit order </h3>

[待翻译]







<h1 id="Ethernet functional description">Ethernet functional description: DMA controller operation</h1>

DMA有独立的发送和接收引擎，以及 CSR[控制/状态寄存器] 空间。发送引擎将数据从系统内存传输到Tx FIFO，而接收引擎将数据从Rx FIFO传输到系统内存。控制器利用描述符有效地将数据从源移动到目的地，而CPU的干预最小。DMA是为面向包的数据传输而设计的，如以太网中的帧。控制器可配置成在帧发送完成，帧接收完成以及其他正常/错误情况下进行中断操作。DMA和STM32F4xx通过两种数据结构进行通信：

-  Control and status registers (CSR)
-  Descriptor lists and data buffers.[描述符列表和数据缓存]

DMA发送器从STM32F4xx内存的接收buffer中接收数据帧，从STM32F4xx内存的发送buffer中发送数据帧。驻留在STM32F4xx内存中的描述符充当指向这些buffer的指针。

内存区有两个描述符列表：一个用于接收，另一个用于发送。每个列表的基址分别写入DMA Registers 3 and 4。描述符列表被前向链接（隐式或显式）。最后一个描述符可以指向第一个条目用于创建环结构。

描述符的显式链是通过配置第二个地址来实现的，其锁存在接收描述符和发送描述符（RDES1[14]和TDES0[20]）中。

描述符列表驻留在主机的物理内存空间中。每个描述符最多可以指向两个缓冲区。这样就可以使用两个物理寻址的缓冲区，而不是内存中的两个连续的缓冲区。

数据缓冲区位于主机的物理内存空间中，由整个帧或帧的一部分组成，但不能超过单个帧。缓冲区只包含数据。缓冲区状态在描述符中保持。数据链是指跨越多个数据缓冲区的帧。但是，一个描述符不能跨越多个帧。当检测到帧结束时，DMA跳到下一帧缓冲区。数据链可以呗启用或禁用。描述符环和链结构如下图所示。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/Descriptor%20ring%20and%20chain%20structure.jpg"></div>

<h3 id="Initialization of a transfer using DMA"> Initialization of a transfer using DMA </h3>

初始化MAC遵循以下步骤：

1. 通过写入ETH_DMABMR寄存器去设置STM32F4xx总线访问参数。
2. 通过写入ETH_DMAIER寄存器去屏蔽不必要的中断事件。
3. 软件驱动创建发送-接收描述符列表。然后它同时写入ETH_DMARDLAR和ETH_DMATDLAR寄存器，向DMA提供每个列表的起始地址。
4. 通过写入MAC寄存器1、2、3来选择所需的筛选选项。
5. 通过写入MAC ETH_MACCR寄存器去配置和使能发送-接收操作模式。PS和DM位基于自动协商结果设置。
6. 通过写入ETH_DMAOMR寄存器设置bit 13和bit 1并开始传输和接收。
7. 发送和接收引擎进入运行状态，并尝试从各自的描述符列表中获取描述符。然后，接收和发送引擎开始处理接收和发送操作。发送和接收过程彼此独立，可以单独启动或停止。

<h3 id="Host bus burst access"> Host bus burst access </h3>

如果配置ETH_DMABMR中的FB位，DMA会尝试在AHB主接口上执行固定长度的突发传输。最大突发长度由PBL字段指示和限制（ETH_DMABMRR[13:8]）。对于要读取的16个字节，接收和发送描述符总是以最大可能的突发大小（受PBL限制）访问。

仅当发送FIFO中有足够的空间来容纳配置的突发或直到帧结束为止的字节数（小于配置的突发长度）时，传输DMA才启动数据传输。DMA指示起始地址和到AHB主接口所需的传输次数。将AHB接口配置为固定长度突发时，它将使用INCR4，INCR8，INCR16和SINGLE事务的最佳组合传输数据。否则（没有固定长度的脉冲串），它将使用INCR（未定义的长度）和SINGLE事务传输数据。

在接收FIFO中有足够用于配置突发的数据，或者当在接收FIFO中检测到帧结束时（当它小于配置的突发长度时），接收DMA才启动数据传输。

DMA指示AHB主接口所需的起始地址和传输数量。当AHB接口配置为固定长度突发时，它使用INCR4、INCR8、INCR16和单个事务的最佳组合来传输数据。

如果在AHB接口上的固定突发结束之前到达帧结束，则执行虚拟传输以完成固定长度突发。否则重置ETH DMABMR中的FB位，它使用INCR（未定义长度）和SINGLE事务传输数据。

当AHB接口被配置为地址对齐节拍，两个DMA引擎都确保AHB发起的第一个突发传输小于或等于配置的PBL的大小。因此，所有后续的节拍从与配置的PBL对齐的地址开始。DMA只能将节拍的地址上调到16（因为PBL>16），因为AHB接口不支持超过INCR16。

<h3 id="Host data buffer alignment"> Host data buffer alignment </h3>

发送和接收数据缓冲区对起始地址对齐没有任何限制。在我们使用32位内存的系统中，缓冲区的起始地址可以与四个字节中的任何一个对齐。然而，DMA总是使用与总线宽度对齐的地址来启动传输，而不需要字节通道的虚拟数据。这通常发生在以太网帧的开始或结束的传输过程中。

- **Example of buffer read:**

如果发送缓冲区地址为0x00000FF2，并且15个字节需要传输，DMA将从地址0x00000FF0读取5个字，但是，当向发送FIFO传输数据时，多余的字节（前两个字节）将被丢弃或忽略。同样，最后一次传输的最后3个字节也将被忽略。DMA总是确保它将完整的32位数据项传输到传输FIFO，除非它是帧的末尾。

- **Example of buffer write:**

如果接收缓冲区地址为0x00000FF2，并且需要传输接收帧的16个字节，则DMA将从地址0x00000FF0写入5个完整的32位数据项。但第一次传输的前2个字节和第3次传输的最后2个字节将有伪数据。

<h3 id="Buffer size calculations">Buffer size calculations </h3>

DMA不更新发送/接收描述符的字段大小。DMA只更新描述符的状态字段（xDES0），驱动必须计算其尺寸。

发送DMA向MAC核心传输精确的字节数（由TDES1中的buffer size字段表示）。如果描述符被标记为第一个（设置了TDES0中的FS位），则DMA将第一次从缓冲区传输的位置标记为起始帧。如果描述符被标记为最后一个（TDES0中的LS位），那么DMA将从该数据缓冲区的最后一次传输的位置标记为结束帧。

接收DMA将数据传输到缓冲区，直到缓冲区已满或接收到帧结束。如果描述符没有标记为最后一个（RDES0中的LS位），则与描述符对应的缓冲区已满，并且缓冲区中的有效数据量由设置描述符的FS位时的缓冲区大小字段减去数据缓冲区指针偏移量来精确指示。当数据缓冲区指针与数据总线宽度对齐时，偏移量为零。如果描述符标记为最后，则缓冲区可能未满（如RDES1中的缓冲区大小所示）。为了计算这个最终缓冲区中的有效数据量，驱动程序必须读取帧长度（RDES0[29:16]中的FL位）并减去该帧中前面的缓冲区大小之和。接收DMA总是用新的描述符传输下一帧的开始。

即使接收缓冲区的起始地址与系统数据总线宽度不一致，系统也应分配一个与系统总线宽度一致的接收缓冲区。

例如，如果系统从地址0x1000开始分配1024字节（1kb）的接收缓冲区，软件可以 *在接收描述符中对缓冲区起始地址进行编程* ，使其具有0x1002偏移量。接收DMA在前两个位置（0x1000和0x1001）使用伪数据将帧写入该缓冲区。实际帧是从位置0x1002写入的。因此，即使由于起始地址偏移，缓冲区大小被编程为1024字节，该缓冲区中的实际有用空间是1022字节。

<h3 id="DMA arbiter"> DMA arbiter </h3>

DMA内部的仲裁器负责AHB主接口的发送和接收通道访问之间的仲裁。

有两种类型的仲裁：循环仲裁和固定优先级。当选择循环仲裁时（重置ETH_DMABMR中的DA位），当同时发送和接收DMAs请求访问时，仲裁器将按照ETH DMABMR中PM位设置的比率分配数据总线。当设置了DA位时，接收DMA总是比发送DMA具有更高的数据访问优先级。

<h3 id="Error response to DMA"> Error response to DMA </h3>

对于由DMA信道发起的任何数据传输，如果从机应答为错误响应，则DMA停止所有操作并更新状态寄存器（ETH DMASR寄存器）中的错误位和致命总线错误位。DMA控制器只能在软或硬重置外围设备并重新初始化DMA之后才能恢复运行。

<h3 id="Tx DMA configuration"> Tx DMA configuration </h3>

<h5 id="default mode"> TxDMA operation: default (non-OSF) mode </h5>

发送DMA引擎的默认模式如下：

1. 用户在用以太网帧数据设置相应的数据缓冲器之后，设置传输描述符（TDES0-TDES3）并设置自己的位（TDES0[31]）。
2. 一旦设置了ST位（ETH_DMAOMR寄存器[13]），DMA进入运行状态。
3. 在运行状态下，DMA轮询发送描述符列表以查找需要传输的帧。轮询开始后，它将以环式或链式顺序描述符工作。如果DMA检测到描述符被标记为CPU拥有，或者如果发生错误情况，则传输被挂起，并且传输缓冲区不可用（ETH_DMASR寄存器[2]）和正常中断摘要（ETH_DMASR寄存器[16]）都被设置。此时，发送引擎进入步骤9。
4. 如果所获取的描述符被标记为属于DMA（设置了TDES0[31]），则DMA将从所获取的描述符解码发送数据缓冲区地址。
5. DMA从STM32F4xx存储器中获取传输数据并传输数据。
6. 如果一个以太网帧存储在多个描述符的数据缓冲区上，DMA将关闭中间描述符并获取下一个描述符。重复步骤3、4和5，直到以太网帧数据传输结束。
7. 当帧传输完成时，如果为帧启用了IEEE 1588时间戳（如传输状态所示），则时间戳值将写入包含帧结束缓冲区的传输描述符（TDES2和TDES3）。然后将状态信息写入该传输描述符（TDES0）。因为在这个步骤中清除了自己的位，CPU现在拥有这个描述符。如果没有为此帧启用时间戳，DMA不会改变TDES2和TDES3的内容。
8. 传输中断（ETH_DMASR寄存器[0]）是在完成帧的传输后设置的，该帧在其最后一个描述符中设置了完成时中断（TDES1[31]）。
9. 在挂起状态下，DMA在接收到发送轮询请求时尝试重新获取描述符（并因此返回到步骤3），下溢中断状态位被清除。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/TxDMA%20operation%20in%20Default%20mode.jpg"></div>

<h5 id="OSF mode"> TxDMA operation: OSF mode </h5>

在运行状态下，发送过程可以同时获取两个帧，而无需关闭第一帧的状态描述符（如果OSF位设置在ETH DMAOMR寄存器[2]）。当传输过程完成传输第一帧时，它立即轮询第二帧的传输描述符列表。如果第二帧有效，则在写入第一帧的状态信息之前传输该帧。在OSF模式下，运行状态发送DMA按照以下顺序运行：

1. DMA按照TxDMA（默认模式）的步骤1–6中所述进行操作。
2. 在不关闭前一帧的最后一个描述符的情况下，DMA获取下一个描述符。
3. 如果DM获取到描述符，DMA将对该描述符中的发送缓冲地址进行解码。如果DMA不拥有描述符，DMA将进入挂起模式并跳到步骤7。
4. DMA从STM32F4xx内存中获取发送帧并传输该帧，直到结束帧数据被传输，如果该帧被拆分到多个描述符，则关闭中间描述符。
5. DMA等待前一帧的传输状态和时间戳。当状态可用时，且捕获了这样的时间戳（由状态位指示），DMA将时间戳写入TDES2和TDES3。然后，DMA将状态写入相应的TDES0，用清除的当前的位，从而关闭描述符。如果前一帧没有启用时间戳，DMA不会改变TDES2和TDES3的内容。
6. 如果启用，则设置传输中断，DMA获取下一个描述符，然后进入步骤3（状态正常时）。如果先前的传输状态显示下溢错误，DMA将进入挂起模式（步骤7）。
7. 在挂起模式下，如果DMA接收到挂起的状态和时间戳，它会将时间戳（如果对当前帧启用）写入TDES2和TDES3，然后将状态写入相应的TDES0。然后设置相关的中断并返回到挂起模式。
8. DMA只有在接收到传输轮询请求（ETH_DMATPDR寄存器）后才能退出挂起模式并进入运行状态（根据挂起状态转到步骤1或步骤2）。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-STM32F4xxP_Ether/TxDMA%20operation%20in%20OSF%20mode.jpg"></div>

<h3 id="Transmit frame processing"> Transmit frame processing </h3>

发送DMA要求数据缓冲区包含完整的以太网帧，不包括前导码、pad字节和FCS字段。DA、SA和Type/Len字段包含有效数据。

如果发送描述符指示MAC核心必须禁用CRC或pad插入，则缓冲区必须具有完整的以太网帧（不包括前导码），包括CRC字节。

帧可以是数据链，跨越多个缓冲区。帧必须由首描述符（TDES0[28]）和尾描述符（TDES0[29]）分隔。

当传输开始时，必须在第首描述符中设置（TDES0[28]）。此时，帧数据从存储器缓冲器传输到发送FIFO。同时，如果当前帧的尾描述符（TDES0[29]）被清除，则发送过程尝试获取下一个描述符。发送过程期望被清除在这个描述符中（TDES0[28]）。如果TDES0[29]被清除，则表示这是一个中间缓冲区。如果TDES0[29]被设置，则表示这是的最后一个帧的缓冲区。

在最后一个帧的缓冲区被发送之后，DMA写回最终状态信息到发送描述符0(TDES0)，那有发送描述符0中设置的最后一个字段(TDES0[29])。

此时，如果设置了完成时中断（TDES0[30]），则设置传输中断（在ETH DMASR寄存器[0]），则获取下一个描述符，并重复该过程。

实际帧传输在发送FIFO达到可编程传输阈值（ETH DMAOMR寄存器[16:14]）或FIFO中包含完整帧之后开始。存储转发模式也有一个选项（ETH_DMAOMR寄存器[21]）。当DMA完成帧传输时，描述符被释放（自己的位TDES0[31]被清除）。

<h3 id="Transmit polling suspended"> Transmit polling suspended </h3>

发送轮询可在以下任一情况下暂停：

- DMA检测到描述符被CPU拥有（TDES0[31]=0），并传输缓冲区不可用标志被设置（ETH U DMASR寄存器[2]）。想要继续，程序必须将描述符所有权授予DMA，然后发出 **Poll Demand** 命令。

- 当检测到下溢导致的传输错误时，帧传输被中止；对应的发送描述符0（TDES0）位被设置。如果出现第二种情况，则Abnormal Interrupt Summary（在ETH_DMASR寄存器[15]）和Transmit Underflow（在ETH_DMASR寄存器[5]）两个 bits 被设置，并将信息写入发送描述符0，由此导致挂起。如果DMA由于第一个条件而进入挂起状态，则Abnormal Interrupt Summary（ETH_DMASR寄存器[16]）和Transmit Buffer Unavailable（ETH_DMASR寄存器[2]）两个 bits 被设置。在这两种情况下，发送列表中的位置被保留。保留的位置是DMA关闭的最后一个描述符之后的描述符的位置。在纠正暂停原因后，程序必须明确发出发送Transmit Poll Demand 命令。

