- <a href="#DESCRIPTION">概述</a>
- <a href="#CONFIGURATION">配置事项</a>
- <a href="#ARCHITECTURE">结构</a>
- <a href="#HOSTNAME">主机名称</a>
- <a href="#HALT SIGNAL">停止信号</a>
- <a href="#REBOOT SIGNAL">重启信号</a>
- <a href="#INIT COMMAND">初始化命令</a>
- <a href="#INIT WORKING DIRECTORY">初始化工作路径</a>
- <a href="#PROC">PROC</a>
- <a href="#EPHEMERAL ">EPHEMERAL</a>
- <a href="#NETWORK">网络</a>
- <a href="#NEW PSEUDO TTY INSTANCE">新的伪TTY实例 (DEVPTS)</a>
- <a href="#CONTAINER SYSTEM CONSOLE">容器系统控制台</a>
- <a href="#CONSOLE DEVICES LOCATION">控制台设备位置</a>
- <a href="#DEV DIRECTORY">/DEV 目录</a>
- <a href="#MOUNT POINTS">挂载点</a>
- <a href="#ROOT FILE SYSTEM">根文件系统</a>
- <a href="#CONTROL GROUP">控制组</a>


<h2 id="DESCRIPTION">概述</h2>

LXC 支持非特权容器。非特权容器是指在没有任何特权的情况下运行的容器。这需要在运行容器的内核中支持 namespaces 。使用 namespaces 合并到主线内核后 LXC 第一时间支持非特权容器的运行。

本质上，用户 namespaces  隔离给定的 uid 和 gid 集。这是通过主机上一系列 uid 和 gid 与容器（非特权）中不同的uid 和 gid 之间建立映射来实现的。内核将以这样一种方式转换这个映射：在容器中，所有 uid 和 gid 都会出现在主机上，而在主机上，这些 uid 和 gid 实际上没有特权。例如，容器中以 UID 和 GID 为 0 的身份运行在主机上的进程可能显示为 UID 和 GID 为 100000。实际操作的细节可以从相应的 namespaces 手册页收集。UID 和 GID 映射可以用关键词 lxc.idmap 来定义。

Linux 容器是用一个简单的配置文件定义的。配置文件中的每个选项在一行中都有固定格式： key=value。 “#” 字符表示该行是注释。列表选项（如 capabilities 和 cgroups 选项）可以在没有值的情况下使用，以该字符清除该选项以前定义的任何值。

LXC namespaces 配置键使用单点。这意味着复杂的配置键，例如 lxc.net.0 展开各种子项：例如 lxc.net.0.type, lxc.net.0.link, lxc.net.0.ipv6.address ，以及其他更细参数的配置。


<h2 id="CONFIGURATION">配置事项</h2>

为了简化对多个相关容器的管理，可以使用一个容器配置文件来加载另一个文件。例如，网络配置可以在由多个容器包含的一个公共文件中定义。然后，如果将容器移动到另一个主机，则可能只需要更新一个文件。

```
lxc.include = /file path
```

指定要包含的文件。包含的文件必须采用相同的有效 lxc 配置文件格式。


<h2 id="ARCHITECTURE">结构</h2>

允许设置容器的体系结构。例如，为在 64 位主机上运行 32 位二进制文件的容器设置 32 位体系结构。这修复了容器脚本，这些脚本依赖于体系结构来完成一些工作，比如下载包。

```
lxc.arch = parameter
```

指定容器的体系结构，一些有效的选项：x86, i686, x86_64, amd64


<h2 id="HOSTNAME">主机名称</h2>

utsname 部分定义为容器所设置的主机名。这意味着容器可以设置自己的主机名，而无需更改系统中的主机名。这使得容器的主机名是私有的。

```
lxc.uts.name = parameter
```

<h2 id="HALT SIGNAL">停止信号</h2>

为了彻底关闭容器，允许指定信号名称或编号发送到容器的 init 进程。不同的 init 系统可以使用不同的信号来执行干净的关闭顺序。此选项允许以 kill（1）方式指定信号，例如 SIGPWR、SIGRTMIN+14、SIGRTMAX-10 或普通数字。默认信号是SIGPWR。

```
lxc.signal.halt = parameter
```

<h2 id="REBOOT SIGNAL">重启信号</h2>

允许指定信号名称或编号用作重新启动容器。此选项允许以kill（1）方式指定信号，例如 SIGTERM、SIGRTMIN+14、SIGRTMAX-10 或普通数字。默认信号是 SIGINT。

```
lxc.signal.reboot = parameter
```

<h2 id="STOP SIGNAL">停止信号</h2>

允许指定信号名称或编号用作强制关闭容器。此选项允许以kill（1）方式指定信号，例如 SIGKILL、SIGRTMIN+14、SIGRTMAX-10 或普通数字。默认信号是 SIGKILL。

```
lxc.signal.stop = parameter
```

<h2 id="INIT COMMAND">初始化命令</h2>

设置此命令用作初始化容器的操作。

```
lxc.execute.cmd
```

这是从容器 rootfs 的绝对路径到二进制文件的默认执行方式，这与 lxc-execute 是相等的。

```
lxc.init.cmd = /sbin/init.
```

这是从容器 rootfs 的绝对路径到二进制文件的初始化方式。这与 lxc-start 是等同的。默认值是 /sbin/init.


<h2 id="INIT WORKING DIRECTORY">初始化工作路径</h2>

将容器内的绝对路径设置为容器的工作目录。在执行 init 之前，LXC 将切换到这个目录。

```
lxc.init.cwd 
```

容器中用作工作目录的绝对路径。


<h2 id="INIT ID">初始化ID</h2>

设置用于 init 系统和后续命令的 UID/GID。请注意，由于缺少权限，在引导系统容器时使用非根 UID 可能无法工作。在运行应用程序容器时，设置 UID/GID 非常有用。默认为：UID（0），GID（0）

```
lxc.init.uid = parameter
lxc.init.gid = parameter
```


<h2 id="PROC">PROC</h2>

配置容器 pro 文件系统。

```
lxc.proc.[proc file name]
```

指定要设置的 proc 文件名。可用的文件名是 /proc/PID/ 列出的文件名。例如：

```
lxc.proc.oom_score_adj = 10
```


<h2 id="EPHEMERAL">EPHEMERAL</h2>

允许指定容器是否在关闭时销毁。

```
lxc.ephemeral = parameter
```

唯一允许的值是 0 和 1。将此设置为 1 可在关闭时销毁容器。


<h2 id="NETWORK">网络</h2>

network 部分定义如何在容器中虚拟化网络。网络虚拟化作用于第二层。为了使用网络虚拟化，必须指定参数来定义容器的网络接口。即使系统只有一个物理网络接口，也可以在容器中分配和使用多个虚拟接口。

## lxc.net

可在没有值的情况下使用，以清除所有以前的网络选项。

## lxc.net.[i].type

这行命令指定要用于容器的网络虚拟化类型。

通过在所有键 lxc.net.* 之后使用附加索引 i 可以指定多个网络。例如， lxc.net.0.type = veth 和 lxc.net.1.type = veth 是指定相同类型的两个不同网络。共享同一索引 i 的所有密钥将被视为属于同一网络。例如：lxc.net.0.link = br0 属于 lxc.net.0.type

目前，不同的虚拟化类型可以是：

**none：** 将导致容器共享主机的网络 namespace 。这意味着主机网络设备可以在容器中使用。这也意味着，如果容器和主机都有 upstart 作为 init，容器中的 'halt' 将关闭主机。请注意，由于无法装载 sysfs，未经授权的容器无法使用此设置。一个不安全的解决方法是绑定挂载主机的 sysfs。

**empty:** 将只创建回环接口。

**veth：** 创建一个虚拟以太网给设备，其中一端分配给容器，另一端分配给主机。lxc.net.[i].veth.mode 指定父级将在主机上使用的模式。可接受的模式是(bridge)网桥和路由器(router)。如果未指定，则模式默认为桥接。在网桥模式下，主机端根据 lxc.net.[i].link 选项连接到指定的网桥。如果未指定网桥链路，则将创建 veth pair 设备，但不会连接到任何网桥。否则，必须在启动容器之前在系统上创建网桥。lxc 不会处理容器之外的任何配置。在路由器模式下，容器的 IP 地址会被创建，并指向主机端 veth 接口。此外，主机端 veth 接口上添加代理 ARP 和代理 NDP 条目，用于容器中定义的网关 IP，以允许容器到达主机。默认情况下，lxc 为属于容器外部的网络设备选择一个名称，但是如果您希望自己处理这个名称，可以告诉 lxc 使用 lxc.net.[i].veth.pair 选项（出于安全原因忽略此选项的非特权容器除外）。可以使用  lxc.net 在主机上添加指向容器的静态路由。

[i].veth.ipv4.route 和 lxc.net.[i].veth.ipv6.route 选项。多条线指定多个路由。路线的格式为 x.y.z.t/m 例如 192.168.1.0/24

在网桥模式下，可以使用lxc.net.[i].veth.vlan.id 选项设置未标记的VLAN成员身份。它接受一个特殊值 'none' ，表示应该从网桥的默认未标记 VLAN 中删除容器端口。lxc.net.[i].veth.vlan.tagged.id 选项可以被多次指定，以将容器的网桥端口成员身份设置为一个或多个标记的 VLAN。

**vlan：** vlan 接口与 lxc.net.[i].link 指定的接口链接并分配给容器。vlan标识符是用选项 lxc.net.[i].vlan.id 指定的。

**macvlan：** macvlan 接口与 lxc.net.[i].link 指定的接口链接并分配给容器。lxc.net.[i].macvlan.mode 指定 macvlan 用于相同的上层设备里不同 macvlan 间通信的模式。可接受的模式有 private、vepa、bridge 和 passthru。在专用模式下，该设备从不与同一上层设备上的任何其他设备通信（默认）。在vepa模式下，新的虚拟以太网端口聚合器（vepa）模式假定相邻网桥返回的源和目标都在 macvlan 端口本地的所有帧上，例如网桥被设置成反射式中继器。在 VEPA 模式下，来自 upper_dev 的广播帧被广播到所有 macvlan 接口，本地帧不在本地传送。在网桥模式下，它提供同一端口上不同 macvlan 接口之间的简单网桥行为。从一个接口到另一个接口的帧是直接传送的，而不是从外部发送出去。广播帧被广播到所有其他的网桥端口和外部接口，但是当它们从反射中继器中继返回时，我们不会再次发送它们。因为我们知道所有的 MAC 地址，macvlan 网桥模式不需要像网桥模块那样学习或 STP。在 passthru 模式下，物理接口接收到的所有帧都被转发到 macvlan 接口。一个物理接口只能有一个处于 assthru 模式的 macvlan 接口。

**ipvlan：** ipvlan 接口与 lxc.net.[i].link 指定的接口链接并分配给容器。lxc.net.[i].ipvlan.mode 指定ipvlan 用于相同的上策设备里不同的 ipvlan 间通信的模式。可接受的模式有 l3、l3s、和l2。默认是 l3 模式。在 L3 模式下，连接到相关设备的堆栈实例上会发生最多 L3 TX 处理，数据包被切换到父设备的堆栈实例进行 L2 处理，并且在数据包在出站设备上排队之前，将使用来自该实例的路由。在此模式下，从设备将不接收也不能发送多播/广播通信。在 L3 模式下，TX 处理与 L3 模式非常相似，只是 iptables（conn tracking）在这种模式下工作，因此它是三级对称（L3s）。

**phys：** 已经存在的接口由 lxc.net.[i].link 指定分配给容器。


## lxc.net.[i].flags

指定要执行的网络操作。

up：激活接口。

## lxc.net.[i].link

指定用于实际网络流量的接口。

## lxc.net.[i].l2proxy

控制是否将第 2 层 IP 邻居代理项添加到容器的 IP 地址的 lxc.net.[i].link 接口。可以设置为 0 或 1，默认值为 0。与 IPv4 地址一起使用时，需要设置以下 sysctl 值：net.ipv4.conf.[link].forwarding=1 当与 IPv6 地址一起使用时，需要设置以下 sysctl 值：net.ipv6.conf.[link].proxy_ndp=1 net.ipv6.conf.[link].forwarding=1

## lxc.net.[i].mtu

指定此接口的最大传输单位。

## lxc.net.[i].name

接口名称是动态分配的，但是如果由于容器使用的配置文件使用通用名称（例如 eth0）而需要另一个名称，则此选项将重命名容器中的接口。

## lxc.net.[i].hwaddr

默认情况下，接口mac地址是动态分配给虚拟接口的，但在某些情况下，这是解决 mac 地址冲突或始终具有相同的链路本地ipv6 地址所必需的。地址中的任何 “x” 都将被随机值替换，这在模板中是允许被设置的。

## lxc.net.[i].ipv4.address

指定要分配给虚拟化接口的 ipv4 地址。几行指定了几个 ipv4 地址。地址的格式为 x.y.z.t/m，例如 192.168.1.123/24。

## lxc.net.[i].ipv4.gateway

指定要用作容器内网关的 ipv4 地址。地址的格式是 x.y.z.t，例如 192.168.1.123。也可以具有特殊值 auto，这意味着从网桥接口获取主地址（由 lxc.net.[i].link 选项）并将其用作网关。仅当使用veth、macvlan 和 ipvlan 网络类型时，auto 才可用。也可以具有特殊值 dev，这意味着将默认网关设置为设备路由。这主要用于第 3 层网络模式，如 IPVLAN。

## lxc.net.[i].ipv6.address

指定要分配给虚拟化接口的ipv6地址。几行指定了几个 ipv6 地址。地址的格式是 x::y/m，例如 2003:db8:1:0:214:1234:fe0b:3596/64

## lxc.net.[i].ipv6.gateway

指定要用作容器内网关的 ipv6 地址。地址的格式是 x::y，例如 2003:db8:1:0::1 也可以有特殊值 auto，这意味着从网桥接口获取主地址（由 lxc.net.[i].link 选项）并将其用作网关。仅当使用 veth、macvlan 和 ipvlan 网络类型时，auto 才可用。也可以具有特殊值 dev，这意味着将默认网关设置为设备路由。这主要用于第 3 层网络模式，如 IPVLAN。

## lxc.net.[i].script.up

添加一个配置选项，在创建和配置从主机端使用的网络后，用于指定要执行的脚本。除了适用于所有挂钩的信息，以下信息也提供给脚本：

- LXC_HOOK_TYPE：钩子类型，这不是 “up” 就是 “down”。
- LXC_HOOK_SECTION: 'net' 类型区域。
- LXC_NET_TYPE: 网络类型，这是此处列出的有效网络类型之一（例如 vlan、macvlan、ipvlan、veth）。
- LXC_NET_PARENT: 主机上的父设备，这只为网络类型 mavclan、veth、phys 设置。
- LXC_NET_PEER: 主机上对等设备的名称，这只为 “veth” 网络类型设置。请注意，此信息仅在以下情况下可用 lxc.hook.version 版设置为 1。

以上信息是以环境变量的形式提供还是作为脚本的参数提供，取决于 lxc.hook.version 的值. 如果设置为1，则以环境变量的形式提供信息。如果设置为 0，则作为脚本的参数提供信息。

脚本的标准输出在调试级别记录，标准错误不会被记录，但是可以通过钩子将其标准错误重定向到标准输出来捕获。

## lxc.net.[i].script.down

在销毁从主机端使用的网络之前，添加一个配置选项去指定一个脚本执行。除了适用于所有挂钩的信息，以下信息也提供给脚本：

- LXC_HOOK_TYPE：钩子类型，这不是 “up” 就是 “down”。
- LXC_HOOK_SECTION: 'net' 类型区域。
- LXC_NET_TYPE: 网络类型，这是此处列出的有效网络类型之一（例如 vlan、macvlan、ipvlan、veth）。
- LXC_NET_PARENT: 主机上的父设备，这只为网络类型 mavclan、veth、phys 设置。
- LXC_NET_PEER: 主机上对等设备的名称，这只为 “veth” 网络类型设置。请注意，此信息仅在以下情况下可用 lxc.hook.version 版设置为 1。

以上信息是以环境变量的形式提供还是作为脚本的参数提供，取决于 lxc.hook.version 的值. 如果设置为1，则以环境变量的形式提供信息。如果设置为 0，则作为脚本的参数提供信息。

脚本的标准输出在调试级别记录，标准错误不会被记录，但是可以通过钩子将其标准错误重定向到标准输出来捕获。


<h2 id="NEW PSEUDO TTY INSTANCE">新的伪TTY实例 (DEVPTS)</h2>

为了实现更严格的隔离，容器可以有自己的伪 tty 私有实例。

**lxc.pty.max**

如果设置，容器将有一个新的伪 tty 实例，使其私有化。该值指定 pty 实例允许的最大伪 tty 数（此限制尚未实现）。


<h2 id="CONTAINER SYSTEM CONSOLE">容器系统控制台</h2>

<h2 id="CONSOLE THROUGH THE TTYS">控制台穿透TTYS</h2>

<h2 id="CONSOLE DEVICES LOCATION">控制台设备位置</h2>

<h2 id="DEV DIRECTORY">/DEV 目录</h2>

默认情况下，lxc 会在容器的 /dev 目录中创建一些符号链接（fd、stdin、stdout、stderr），但不会自动创建设备节点条目。这允许用户根据需要在容器 rootfs 中设置容器的 /dev。如果 lxc.autodev 设置为 1，那么在安装容器的 rootfs 之后，LXC 将在 /dev 下挂载一个新的 tmpfs（默认限制为500K，除非 lxc 中另有规定）并填充一组最小的初始设备。启动一个容器时，它包含基于 “systemd” 的 “init” ，这通常是必需的，但在其他时候可能是可选的。通过使用 lxc.hook.autodev 钩子可以在 containers/dev 目录中创建其他设备。

**lxc.autodev：**  将其设置为 0，用来阻止 LXC 在启动容器时装载和填充最小的 /dev。

**lxc.autodev.tmpfs.size：** 设置它以定义 /dev tmpfs 的大小。默认值为 500000（500K）。如果使用了参数但没有值，则使用默认值。

<h2 id="MOUNT POINTS">挂载点</h2>

<h2 id="ROOT FILE SYSTEM">根文件系统</h2>

容器的根文件系统可以不同于主机系统的根文件系统。

**lxc.rootfs.path**

指定容器的根文件系统。它可以是图像文件、目录或块设备。如果未指定，容器将与主机共享其根文件系统。对于目录或简单块设备支持的容器，可以使用路径名。如果 rootfs 由 nbd(network block device) 设备支持，那么 nbd:file:1 指定文件应附加到 nbd 设备，分区 1 应作为 rootfs 挂载。overlayfs:/lower:/upper 指定 rootfs 为 overlay，其中 /upper 被以读写方式安装在 /lower 的只读装载上。对于 overlay ，可以指定多个 /lower 的目录。loop:/file 告诉 lxc 将 /file 附加到循环设备并挂载循环设备。

**lxc.rootfs.mount**

**lxc.rootfs.options**

**lxc.rootfs.managed**

<h2 id="CONTROL GROUP">控制组</h2>

控制组部分包含不同子系统的配置。lxc 不检查子系统名称的正确性。这样做的缺点是在容器启动之前不会检测到配置错误，但是它的优点是允许任何新建的子系统。

## lxc.cgroup.[controller name]

继承等级的 cgroup 上设置的控制组值。控制器名称是控制组的名称。允许的名称及其值的语法不是由 LXC 指定的，而是取决于启动容器时运行的 Linux 内核的特性，例如 lxc.cgroup.cpuset.cpus

## lxc.cgroup2.[controller name]

统一的 cgroup 上设置的控制组值。控制器名称是控制组的名称。允许的名称及其值的语法不是由 LXC 指定的，而是取决于启动容器时运行的 Linux 内核的特性，例如  lxc.cgroup2.memory.high

## lxc.cgroup.dir

指定将在其中创建容器的 cgroup 的目录或路径。例如，设置 lxc.cgroup.dir = my cgroup/first 对于名为 “c1” 的容器，将创建容器的 cgroup 作为 “my cgroup” 的子 cgroup。例如，如果用户的当前 cgroup “my user” 位于 cgroup v1层次结构中 cpuset 控制器的根 cgroup 中，则会为容器创建 cgroup “/sys/fs/cgroup/cpuset/my-user/my-cgroup/first/c1”。任何丢失的 cgroup 都将由 LXC 创建。这假定用户对其当前 cgroup 具有写访问权限。

## lxc.cgroup.dir.container

这与 lxc.cgroup.dir 相似，但必须与 lxc.cgroup.dir.monito 一起使用，且只影响容器的 cgroup 路径。此选项与 lxc.cgroup.dir 互斥。 值得注意的是，容器附加到的最终路径可以通过 lxc.cgroup.dir.container.inner 选项进一步扩展。

## lxc.cgroup.dir.monitor

这是对应于 lxc.cgroup.dir.container 的监视进程。 

## lxc.cgroup.dir.monitor.pivot

## lxc.cgroup.dir.container.inner

## lxc.cgroup.relative

<h2 id="CAPABILITIES">功能</h2>

如果这个功能是以 root 用户身份运行的，那么可以将这些功能放到容器中。

**lxc.cap.drop**

指定要放入容器中的功能。允许用空格分隔定义多个功能的单行线。格式是能力定义的小写，不带 “CAP_” 前缀，例如，CAP_SYS_MODULE 应该被指定为 sys_module

见 capabilities(7)，如果没有使用任何值，lxc 将清除到目前为止要丢弃的功能。

**lxc.cap.keep**

指定要保存在容器中的功能，所有其他功能都将被放弃。当遇到特殊值 “none” 时，lxc 将清除在此之前指定的任何保持功能。“none” 可单独用于删除所有功能。

<h2 id="SECCOMP CONFIGURATION">SECCOMP配置</h2>

容器可以通过在启动时加载 seccomp 配置文件，减少可用系统调用的集合。seccomp 配置文件开头格式为：第一行为版本号，第二行为策略类型，然后是配置。

当前支持版本 1 和版本 2。在版本 1 中，策略是一个简单的 allowlist。因此，第二行必须读为 “allowlist” ，文件的其余部分每行包含一个（数字）syscall 编号。每个 syscall 编号都是 allowlisted 的，而每个未列出的编号都被denylisted 以便在容器中使用。

在版本 2 中，策略可以是 denylist 或 allowlist，支持每个规则和每个策略的默认操作，并支持从文本名称按体系结构进行系统调用解析。

denylist 策略的一个示例，在该策略中，除了 mknod 之外，所有系统调用都是允许的，mknod 将不执行任何操作并返回 0（success），例如：

```
2
denylist
mknod errno 0
ioctl notify
```

将 “errno” 指定为 action 将导致 LXC 注册一个 seccomp 筛选器，该筛选器将导致向调用方返回特定的 errno。可以在 “errno” 的动作字之后指定 errno 值。

指定 “notify” 作为操作，将导致 LXC 注册 seccomp 侦听器，并从内核检索侦听器文件描述符。进行系统调用时，注册为“notify” 的内核将生成轮询事件，并通过文件描述符发送消息。调用者可以读取此消息，检查包括参数在内的系统调用。基于这些信息，调用者应该发回一条消息，通知内核要采取的操作。在该消息被发送之前，内核将阻止调用进程。要读取和发送的消息格式记录在 seccomp 中。

## lxc.seccomp.profile

指定一个包含 seccomp 配置的文件，并在容器开启之前加载。

## lxc.seccomp.allow_nesting

## lxc.seccomp.notify.proxy

## lxc.seccomp.notify.cookie