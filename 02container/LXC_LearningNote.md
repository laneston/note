<a href="#Container overview">容器概述</a>
- <a href="#Container and virtual machine">容器与虚拟机</a>
- <a href="#Why use containers">为什么要用容器</a>
- <a href="#What is LXC">什么是LXC</a>

<a href="#Namespace">Namespace</a>
- <a href="#Types of namespace">Namespace的种类</a>
- <a href="#Use of namespace">Namespace的使用</a>
- <a href="#Notes on using namespace">Namespace使用注意事项</a>
- <a href="#Functions and features of namespace">Namespace的功能和特性</a>

<a href="#Cgroups">Cgroups</a>
- <a href="#what is Cgroups">什么是Cgroups</a>
- <a href="#the function of Cgroups">Cgroups的功能</a>
- <a href="#Introduction to subsystem">子系统简介</a>
- <a href="#How to manage control group">如何管理控制群组</a>

<h1 id="Container overview">容器概述</h1>

<h3 id="Container and virtual machine">容器与虚拟机</h3>

容器是一种基于操作系统层级的虚拟化技术。“容器”是轻量级的虚拟化技术，与我们常用的 VMware 虚拟机不同，因为 LXC 不仿真硬件，且由于容器与主机共享相同的操作系统，共用相同的硬件资源，而虚拟机是寄生在宿主系统上的软件，与宿主系统或其它寄生系统抢占硬件的资源。

<h3 id="Why use containers">为什么要用容器</h3>

既然虚拟机与容器都能提供软件所需的执行环境，那容器的存在又有什么必要性呢？

容器是操作系统层级的虚拟化技术，与传统的硬件抽象层的虚拟化技术相比有以下优势：

1. 更小的虚拟化开销，虚拟机模拟硬件和操作系统，但是容器只模拟操作系统，因此更轻量级、速度更快。
2. 更快的部署。利用容器来隔离特定应用，只需安装容器，即可使用容器相关命令来创建并启动容器来为应用提供虚拟执行环境。传统的虚拟化技术则需要先创建虚拟机，然后安装系统，再部署应用。

<h3 id="What is LXC">什么是LXC</h3>

LXC 是 Linux Containers 的简称，LXC 允许你在宿主操作系统内的容器运行应用。容器在网络、行为等方面都与宿主OS都隔离。LXC 的仿真（模拟）是通过 Linux 内核的 cgroups 和 namespaces 来实现的，因此 LXC 只能模拟基于 Linux 类的操作系统。

<h1 id="Namespace">Namespace</h1>

namespace 是 Linux 内核用来隔离内核资源的方式。通过 namespace 可以让一些进程只能看到与自己相关的一部分资源，而另外一些进程也只能看到与它们自己相关的资源，这两拨进程根本就感觉不到对方的存在。具体的实现方式是把一个或多个进程的相关资源指定在同一个 namespace 中。Linux namespaces 是对全局系统资源的一种封装隔离，使得处于不同 namespace 的进程拥有独立的全局系统资源，改变一个 namespace 中的系统资源只会影响当前 namespace 里的进程，对其他 namespace 中的进程没有影响。

<h3 id="Types of namespace">Namespace的种类</h3>

目前Linux中提供了六类系统资源的隔离机制，分别是：

- Mount: 隔离文件系统挂载点；
- UTS:   隔离主机名和域名信息；
- IPC:   隔离进程间通信；
- PID:   隔离进程的ID；
- Network: 隔离网络资源；
- User:  隔离用户和用户组的ID。

<h3 id="Use of namespace">Namespace的使用</h3>

涉及到 Namespace 的操作口包括 clone(), setns(), unshare() 以及还有 /proc 下的部分文件。为了使用特定的 Namespace，在使用这些接口的时候需要指定以下一个或者多个参数：

- CLONE_NEWNS: 用于指定Mount Namespace
- CLONE_NEWUTS: 用于指定UTS Namespace
- CLONE_NEWIPC: 用于指定IPC Namespace
- CLONE_NEWPID: 用于指定PID Namespace
- CLONE_NEWNET: 用于指定Network Namespace
- CLONE_NEWUSER: 用于指定User Namespace

**clone函数**

可以通过 clone 系统调用来创建一个独立 Namespace 的进程，它的函数描述如下：

```
int clone(int (*child_func)(void *), void *child_stack, int flags, void *arg);
```

它通过 flags 参数来控制创建进程时的特性，比如新创建的进程是否与父进程共享虚拟内存等。比如可以传入 CLONE_NEWNS 标志使得新创建的进程拥有独立的 Mount Namespace，也可以传入多个flags使得新创建的进程拥有多种特性，比如：

```
flags = CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC;
```

传入这个 flags 那么新创建的进程将同时拥有独立的 Mount Namespace、UTS Namespace 和 IPC Namespace。

**通过/proc文件查看已存在的Namespace**

输入 ps -A 即可查看所有程序的 PID

如果需要查看 PID 为 1988 的进程的文件信息，则只需输入以下 ls 命令即可查看对应的信息。

```
sudo ls -al /proc/1988/ns/

lrwxrwxrwx 1 lance lance 0 11月  3 16:39 cgroup -> 'cgroup:[4026531835]'
lrwxrwxrwx 1 lance lance 0 11月  3 16:39 ipc -> 'ipc:[4026531839]'
lrwxrwxrwx 1 lance lance 0 11月  3 16:39 mnt -> 'mnt:[4026531840]'
lrwxrwxrwx 1 lance lance 0 11月  3 16:39 net -> 'net:[4026532000]'
lrwxrwxrwx 1 lance lance 0 11月  3 16:39 pid -> 'pid:[4026531836]'
lrwxrwxrwx 1 lance lance 0 11月  3 16:39 pid_for_children -> 'pid:[4026531836]'
lrwxrwxrwx 1 lance lance 0 11月  3 16:39 user -> 'user:[4026531837]'
lrwxrwxrwx 1 lance lance 0 11月  3 16:39 uts -> 'uts:[4026531838]'
```

**setns函数**

setns() 函数可以把进程加入到指定的 Namespace 中，它的函数描述如下：

```
int setns(int fd, int nstype);
```

它的参数描述如下：

- fd：表示文件描述符，前面提到可以通过打开 /proc/$pid/ns/ 的方式将指定的 Namespace 保留下来，也就是说可以通过文件描述符的方式来索引到某个 Namespace。
- nstype：用来检查 fd 关联 Namespace 是否与 nstype 表明的 Namespace 一致，如果填 0 的话表示不进行该项检查。

**unshare函数**

unshare() 系统调用函数用于将当前进程和所在的 Namespace 分离，并加入到一个新的 Namespace 中，相对于 setns() 系统调用来说，unshare() 不用关联之前存在的 Namespace，只需要指定需要分离的 Namespace 就行，该调用会自动创建一个新的 Namespace。

unshare()的函数描述如下：

```
int unshare(int flags);
```

其中 flags 用于指明要分离的资源类别，它支持的 flag s与 clone 系统调用支持的 flags 类似，这里简要的叙述一下几种标志：

- CLONE_FILES: 子进程一般会共享父进程的文件描述符，如果子进程不想共享父进程的文件描述符了，可以通过这个flag来取消共享;
- CLONE_FS: 使当前进程不再与其他进程共享文件系统信息;
- CLONE_SYSVSEM: 取消与其他进程共享SYS V信号量;
- CLONE_NEWIPC: 创建新的IPC Namespace，并将该进程加入进来。

<h3 id="Notes on using namespace">Namespace使用注意事项</h3>

unshare() 和 setns() 系统调用对 PID Namespace 的处理不太相同，当 unshare PID namespace 时，调用进程会为它的子进程分配一个新的 PID Namespace，但是调用进程本身不会被移到新的 Namespace 中。而且调用进程第一个创建的子进程在新 Namespace 中的PID 为1，并成为新 Namespace 中的 init 进程。

setns()系统调用也是类似的，调用者进程并不会进入新的 PID Namespace，而是随后创建的子进程会进入。

为什么创建其他的 Namespace 时 unshare() 和 setns() 会直接进入新的 Namespace，而唯独 PID Namespace 不是如此呢？

因为调用 getpid() 函数得到的 PID 是根据调用者所在的 PID Namespace 而决定返回哪个 PID，进入新的 PID namespace 会导致 PID 产生变化。而对用户态的程序和库函数来说，他们都认为进程的 PID 是一个常量，PID 的变化会引起这些进程奔溃。

换句话说，一旦程序进程创建以后，那么它的 PID namespace 的关系就确定下来了，进程不会变更他们对应的 PID namespace。

<h3 id="Functions and features of namespace">Namespace的功能和特性</h3>

**Mount Namespace**

Mount Namespace 用来隔离文件系统的挂载点，不同 Mount Namespace 的进程拥有不同的挂载点，同时也拥有了不同的文件系统视图。Mount Namespace 是历史上第一个支持的 Namespace，它通过 CLONE_NEWNS 来标识的。

**挂载的概念**

在Windows下，mount 挂载，就是给磁盘分区提供一个盘符。比如插入U盘后系统自动分配给了它 I:盘符 。这其实就是挂载，退U盘的时候进行安全弹出，其实就是卸载 unmount。

mount 所达到的效果是：像访问一个普通的文件一样访问位于其他设备上文件系统的根目录，也就是将该设备上目录的根节点挂到了另外一个文件系统的页节点上，达到给这个文件系统扩充容量的目的。

可以通过/proc文件系统查看一个进程的挂载信息，具体做法如下：

```
cat /proc/$pid/mountinfo
```

输出格式如下：

|  36   |  35   | 98:0  | /mnt1 | /mnt2 | rw,noatime | master:1 |   -   | ext3  | /dev/root | rw,error=continue |
| :---: | :---: | :---: | :---: | :---: | :--------: | :------: | :---: | :---: | :-------: | :---------------: |
|   1   |   2   |   3   |   4   |   5   |     6      |    7     |   8   |   9   |    10     |        11         |


1. mount ID: 对于 mount 操作一个唯一的ID；
2. parent ID: 父挂载的 mount ID 或者本身的mount ID(本身是挂载树的顶点)；
3. major:minor: 文件关联额设备的主次设备号；
4. root: 文件系统的路径名，这个路径名是挂载点的根；
5. mount point: 挂载点的文件路径名(相对于这个进程的跟目录)；
6. mount options: 挂载选项
7. optional fields: 可选项，格式 tag:value；
8. separator: 分隔符，可选字段由这个单个字符标示结束的；
9. filesystem type: 文件系统类型 type[.subtype]；
10. mount source: 文件系统相关的信息，或者none；
11. super options: 一些高级选项(文件系统超级块的选项)。

**UTS Namespace**

UTS Namespace 提供了主机名和域名的隔离，也就是 struct utsname 里的 nodename 和 domainname 两个字段。不同 Namespace 中可以拥有独立的主机名和域名。那么为什么需要对主机名和域名进行隔离呢？因为主机名和域名可以用来代替IP地址，如果没有这一层隔离，同一主机上不同的容器的网络访问就可能出问题。

**IPC Namespace**

IPC Namespace是对进程间通信的隔离，进程间通信常见的方法有信号量、消息队列和共享内存。IPC Namespace主要针对的是SystemV IPC和Posix消息队列，这些IPC机制都会用到标识符，比如用标识符来区分不同的消息队列，IPC Namespace要达到的目标是相同的标识符在不同的Namepspace中代表不同的通信介质(比如信号量、消息队列和共享内存)。

<h1 id="Cgroups">Cgroups</h1>

<h3 id="what is Cgroups">什么是Cgroups</h3>

Cgroups 是 Linux 内核提供的一种机制，这种机制可以根据特定的行为，把一系列系统任务及其子任务整合（或分隔）到按资源划分等级的不同组内，从而为系统资源管理提供一个统一的框架。Cgroups 可以限制、记录、隔离进程组所使用的物理资源（包括：CPU、memory、IO等），它本质上是系统内核附加在程序上的，为容器实现虚拟化提供的一系列钩子，通过程序运行时对资源的调度触发相应的钩子，从而达到资源追踪和限制的目的。供了基本保证，是构建 Docker 等一系列虚拟化管理工具的基石。

<h3 id="the function of Cgroups">Cgroups的功能</h3>

cgroups 的一个设计目标是为不同的应用情况提供统一的接口，从控制单一进程到操作系统层虚拟化（像OpenVZ，Linux-VServer，LXC）。cgroups 提供：

- 资源限制：组可以被设置不超过设定的内存限制；这也包括虚拟内存。
- 优先级：一些组可能会得到大量的CPU或磁盘IO吞吐量。
- 结算：用来衡量系统确实把多少资源用到适合的目的上。
- 控制：冻结组或检查点和重启动。

Cgroups 最初的目标是为资源管理提供的一个统一的框架，既整合现有的 cpuset 等子系统，也为未来开发新的子系统提供接口。现在的 cgroups 适用于多种应用场景，从单个进程的资源控制，到实现操作系统层次的虚拟化（OS Level Virtualization）。Cgroups 提供了以下功能：

1. 限制进程组可以使用的资源数量（Resource limiting）。比如：memory 子系统可以为进程组设定一个 memory 使用上限，一旦进程组使用的内存达到限额再申请内存，就会触发OOM（out of memory）。
2. 进程组的优先级控制（Prioritization）。比如：可以使用cpu子系统为某个进程组分配特定cpu share。
3. 记录进程组使用的资源数量（Accounting）。比如：可以使用cpuacct子系统记录某个进程组使用的cpu时间
4. 进程组隔离（Isolation）。比如：使用ns子系统可以使不同的进程组使用不同的namespace，以达到隔离的目的，不同的进程组有各自的进程、网络、文件系统挂载空间。
5. 进程组控制（Control）。比如：使用freezer子系统可以将进程组挂起和恢复。

<h3 id="Introduction to subsystem">子系统简介</h3>

- blkio： 这个子系统为块设备设定输入/输出限制，比如物理设备（磁盘，固态硬盘，USB 等等）。
- cpu： 这个子系统使用调度程序提供对 CPU 的 cgroup 任务访问。
- cpuacct： 这个子系统自动生成 cgroup 中任务所使用的 CPU 报告。
- cpuset：这个子系统为 cgroup 中的任务分配独立 CPU（在多核系统）和内存节点。
- devices： 这个子系统可允许或者拒绝 cgroup 中的任务访问设备。
- freezer：这个子系统挂起或者恢复 cgroup 中的任务。
- memory：这个子系统设定 cgroup 中任务使用的内存限制，并自动生成由那些任务使用的内存资源报告。
- net_cls：这个子系统使用等级识别符（classid）标记网络数据包，可允许 Linux 流量控制程序（tc）识别从具体 cgroup 中生成的数据包。
- ns： 名称空间子系统。

<h3 id="How to manage control group">如何管理控制群组</h3>

Cgroup 是透过阶层式的方式来管理的，和程序、子群组相同，都会由它们的 parent 继承部份属性。然而，这两个模型之间有所不同。

### Linux 程序模型

Linux 系统上的所有程序皆为相同 parent 的子程序：init 程序，由 kernel 在开机时执行，并启用其它程序（并且可能会相应地启用它们自己的子程序）。因为所有程序皆源自於单独的父程序，因此 Linux 的程序模型属於单独的阶层或树状目录。
此外，所有除了init以外的程序皆会继承其父程序的环境（例如 PATH 变数）与特定属性（例如开放式的档案描述元）。

### Cgroup 模型

Cgroup 与程序的相似点为：

它们皆属於阶层式，并且子 cgroup 会继承其父群组的特定属性。基础差异就是在同一部系统上，能够同时存在许多不同的 cgroup 阶层。若 Linux 程序模型是个程序的单树状，那麼 cgroup 模型便是个各别、未连接的树状工作（例如程序）。

多重各别的 cgroup 阶层是必要的，因为各个阶层皆连至了「一个或更多」个「子系统」。子系统代表单独的资源，例如 CPU 时间或记忆体。Red Hat Enterprise Linux 6 提供了九个控制群组子系统，以名称和功能列在下方。

Red Hat Enterprise Linux 中的可用子系统:

1. blki 此子系统可设置来自于区块装置（例如像是固态、USB 等等的实体磁碟）的输入/输出存取限制。
2. cpu 此子系统使用了排程器，以提供 CPU cgroup 工作的存取权限。
3. cpuacct 此子系统会自动产生 cgroup 中的工作所使用的 CPU 资源报告。
4. cpuset 此子系统会将个别的 CPU 与内存节点分配给 cgroup 中的工作。
5. devices 此子系统能允许或拒绝控制群组中的任务存取装置。
6. freezer 此子系统可中止或复原控制群组中的工作。
7. memory 此子系统会根据使用於控制群组中的工作的内存资源，自动产生内存报告，然後设定这些工作所能使用的内存限制：
8. net_cls 此子系统会以一个 class 标识符号（classid）来标记网路封包，这能让 Linux 流量控制器（tc）辨识源自於特定控制群组的封包。流量控制器能被配置来指定不同的优先顺序给来自於不同控制群组的封包。
9. ns namespace子系统。