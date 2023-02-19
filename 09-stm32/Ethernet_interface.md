MII即“媒体独立接口”，也叫“独立于介质的接口”。它是IEEE-802.3定义的以太网行业标准，它包括一个数据接口，以及一个MAC和PHY之间的管理接口。RMII全称为“简化的媒体独立接口”，是IEEE-802.3u标准中除MII接口之外的另一种实现。 

# MII

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Ethernet_EVB_Hardware_Note/MII.jpg" width="50%" height="50%"></div>


引脚说明如下：

1. MII_TX_CLK：发送数据使用的时钟信号，对于10M位/s的数据传输，此时钟为2.5MHz，对于100M位/s的数据传输，此时钟为25MHz。
2. MII_RX_CLK：接收数据使用的时钟信号，对于10M位/s的数据传输，此时钟为2.5MHz，对于100M位/s的数据传输，此时钟为25MHz。
3. MII_TX_EN：传输使能信号，此信号必需与数据前导符的起始位同步出现，并在传输完毕前一直保持。
4. MII_TXD[3:0]：发送数据线，每次传输4位数据，数据在MII_TX_EN信号有效时有效。MII_TXD[0]是数据的最低位，MII_TXD[3]是最高位。当MII_TX_EN信号无效时，PHY忽略传输的数据。
5. MII_CRS：载波侦听信号，仅工作在半双工模式下，由PHY控制，当发送或接收的介质非空闲时，使能此信号。 PHY必需保证MII_CRS信号在发生冲突的整个时间段内都保持有效，不需要此信号与发送/接收的时钟同步。 
6. MII_COL：冲突检测信号，仅工作在半双工模式下，由PHY控制，当检测到介质发生冲突时，使能此信号，并且在整个冲突的持续时间内，保持此信号有效。此信号不需要和发送/接收的时钟同步。 
7. MII_RXD[3:0]：接收数据线，每次接收4位数据，数据在MII_RX_DV信号有效时有效。MII_RXD[0]是数据的最低位，MII_RXD[3]是最高位。当MII_RX_EN无效，而MII_RX_ER有效时，MII_RXD[3:0]数据值代表特定的信息(请参考表194)。
8. MII_RX_DV：接收数据使能信号，由PHY控制，当PHY准备好数据供MAC接收时，使能该信号。此信号必需和帧数据的首位同步出现，并保持有效直到数据传输完成。在传送最后4位数据后的第一个时钟之前，此信号必需变为无效状态。为了正确的接收一个帧，有效电平不能滞后于数据线上的SFD位出现。 
9. MII_RX_ER：接收出错信号，保持一个或多个时钟周期(MII_RX_CLK)的有效状态，表明MAC在接收过程中检测到错误。具体错误原因需配合MII_RX_DV的状态及MII_RXD[3:0]的数据值。 
10. 为了产生TX_CLK和RX_CLK时钟信号，外接的PHY模块必需有来自外部的25MHz时钟驱动。该时钟不需要与MAC时钟相同。可以使用外部的25MHz晶体或者GD32F107xx微控制器的MCO引脚提供这一时钟。当时钟来源MCO引脚时需配置合适的PLL，保证MCO引脚输出的时钟为25MHZ。 

# RMII

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Ethernet_EVB_Hardware_Note/RMII.jpg" width="50%" height="50%"></div>

通过将相同的时钟源接到MAC和以太网PHY的REF_CLK引脚保证两者时钟源的同步。可以通过外部的50MHZ信号或者微控制器的MCO引脚提供这一时钟。当时钟来源MCO引脚时需配置合适的PLL，保证MCO引脚输出的时钟为50MHZ。

# RGMII

就是Reduced GMII，GMII采用8位数据传输，RGMII采用4位数据传输，所以这个按字面意思理解就是减少的GMII。这个是因为RGMII在时钟的上升沿和下降沿均采样数据，所以数据位减少一般的情况下还是可以达到千兆的速率。其工作时钟125兆，兼容MII规定的10/100Mbps。

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Ethernet_EVB_Hardware_Note/RGMII.jpg" width="50%" height="50%"></div>

- TX_CLK:  RGMII发送时钟提供125mhz，25mhz，或2.5mhz参考时钟，容差±50 ppm，具体取决于速度。
- TX_CTRL: RGMII发送控制。TX_EN出现在TX_CLK的上升沿。TX_EN和TX_ER出现在TX_CLK的下降沿。
- TXD0..3:  RGMII发送数据。发送数据端口以双速率运行，数据字节的bits[3:0]在TX_CLK的上升沿发送。数据bit[7:4]在TX_CLK的下降沿发送。
- RX_CLK: RGMII 接收时钟提供 125 MHz、25 MHz 或 2.5 MHz 参考时钟，具有 ± 50 ppm 容差，取决于接收数据流的速度。
- RX_CTRL:  RGMII发送控制。RX_DV出现在RX_CLK的上升沿。RX_DV和RX_ER的逻辑导数出现在RX_CLK的下降沿。
- RXD0..3:  RGMII发送数据端。RXD[3:0]运行在双速率模式，在RX_CLK的上升沿发送bits[3:0]，在数据下降沿发送bits[7:4]。
- RESETn:  硬件复位低有效。
- MDC：  串行管理接口的参考时钟。不需要连续的时钟流。最大支持12M。
- MDIO： 串行管理接口的数据。 此管脚需要一个上拉电阻，范围是1.5K-10K.
- CLK125:   125MHZ参考时钟输出。
- *RXDV:  receive data valid ，发送数据有效*
