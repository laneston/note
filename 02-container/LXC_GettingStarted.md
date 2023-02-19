本篇文章记录 LXC 的初步安装与使用的相关内容，目的是为了使读者能够根据本篇内容实现单次安装成功，并能够进行基础的操作内容。

运行平台：
- ubuntu 20.04

## 依赖要求

要使 LXC 成功运行，则需要一个 C 库来支持，支持的 C 库包括: glibc, musl libc, uclib, bionic 

除此之外，还要保证内核的版本大于或等于 2.6.32

查看内核指令：

```
cat /proc/version
```

如果使用 lxc-attach 内核版本要大于等于3.8

如果要使用 unprivileged containers 则：

- 为了 unprivileged CGroups 操作使用 libpam-cgfs 配置你的系统；
- 最新版本的新 uidmap 和新版本的 gidamap；
- Linux 内核大于等于 3.12

推荐的库：

- libcap (to allow for capability drops)
- libapparmor (to set a different apparmor profile for the container)
- libselinux (to set a different selinux context for the container)
- libseccomp (to set a seccomp policy for the container)
- libgnutls (for various checksumming)
- liblua (for the LUA binding)
- python3-dev (for the python3 binding)

## glibc

### 什么是glibc

glibc 的全称是 GUN C Library，这个库为 GUN 系统和 GUN/Linux 系统，以及许多其他使用Linux作为内核的系统提供核心库文件。这些库提供了关键的api，包括isoc11、POSIX.1-2008、BSD、特定于操作系统的api等等。这些api包括诸如open、read、write、malloc、printf、getaddrinfo、dlopen、pthread_create、crypt、login、exit等基础设施。

### 查看glibc

一般情况下，ubuntu 桌面版会自带 glibc 文件。如果要查看版本信息，只需在命令窗口输入：

```
ldd --version
```

## 安装 LXC

使用 ubuntu 10.04 LTS 以上版本，输入以下命令：

```
sudo apt-get install lxc
```

你的系统将有所有的LXC命令可用，它的所有模板以及python3绑定都需要编写LXC脚本。

## 作为用户创建一个非特权容器

非特权容器是最安全的容器，它们使用 uid 和 gid 的映射为容器分配一系列 uid 和 gid

**注：**

*用户标识号(UID):是一个整数，系统内部用它来标识用户。一般情况下它与用户名是一一对应的。如果几个用户名对应的用户标识号是一样的，系统内部将把它们视为同一个用户，但是它们可以有不同的口令、不同的主目录以及不同的登录Shell等。取值范围是0-65535。0是超级用户root的标识号，1-99由系统保留，作为管理账号，普通用户的标识号从100开始。在Linux系统中，这个界限是500。*

*组标识号(GID): 字段记录的是用户所属的用户组。它对应着/etc/group文件中的一条记录。*

这意味着容器中的 uid 0(root) 实质是容器外部的 uid x 类似。因此如果出现严重错误，攻击者设法逃离容器，它们将会发现自己拥有的权限与 nobody 用户一样少。

然而，这也意味着不允许以下常见操作：

1. 大多数文件系统的挂载；
2. 创建设备节点；
3. 针对映射集之外的 uid/gid 的任何操作。

因此，大多数分发模板根本无法使用这些模板。相反，你应该使用“下载”模板，该模板将为您提供已知在这种环境中工作的发行版的预构建映像。

下面说明的前提都是使用最新的 Ubuntu 系统或提供类似体验的替代 Linux 发行版，即最新的内核和 <a href="#shadow">shadow</a> 的最新版本，以及 <a href="#libpam-cgfs">libpam-cgfs</a> 和默认 uid/gid 分配。

首先，需要确保用户在 /etc/subuid 和 /etc/subgid 中定义了 uid 和 gid 映射。在 Ubuntu 系统上，默认分配给系统中的每个新用户 65536 个 uid 和 gid，所以您应该已经有了一个。如果没有，你可以使用用户模式给自己新建一个。

接下来是 /etc/lxc/lxc-usernet，用于为非特权用户设置网络设备配额。默认情况下，不允许用户在主机上创建任何网络设备，若要更改此设置，请添加：

```
your-username veth lxcbr0 10
```

这意味着 “your-username” 是允许创建最多 10 个 veth 设备(virtual Ethernet, 是 Linux 内核支持的一种虚拟网络设备，表示一对虚拟的网络接口，VETH 对的两端可以处于不同的网络命名空间，所以可以用来做主机和容器之间的网络通信。)去连接 lxcbr0 Bridge(lxcbr0 网桥为 Container Station 中的容器提供 Internet 连接)。

**注：**

*Bridge 类似于交换机，用来做二层的交换。可以将其他网络设备挂在 Bridge 上面，当有数据到达时，Bridge 会根据报文中的 MAC 信息进行广播、转发或丢弃。*

完成后，最后一步是创建一个 LXC 配置文件。

1. 如果 ~/.config/lxc 路径不存在，则创建它；
2. 将 /etc/lxc/default.conf 复制到 ~/.config/lxc/default.conf
3. 在文件中增加以下两行：

```
lxc.idmap = u 0 100000 65536
lxc.idmap = g 0 100000 65536
```

值得注意的是，这些值应该与 /etc/subuid 和 /etc/subgid 中的值相匹配，上面的值是标准 Ubuntu 系统上第一个用户所期望的值。

非特权用户只有提前授权一个 cgroup 才能使用非特权容器（cgroup2 授权模型执行此限制，而不是 liblxc）。使用以下系统命令授权 cgroup：

```
systemd-run --unit=myshell --user --scope -p "Delegate=yes" lxc-start <container-name>
```

**注：**

*如果在安装 LXC 之前没有在主机上安装 libpam-cgfs，那么在创建第一个容器之前，您需要确保您的用户属于正确的 cgroup。您可以通过注销并重新登录，或重新启动主机来完成此操作。*

现在，你可以创建第一个容器了：

```
lxc-create -t download -n my-container
```

下载模板将向您显示可供选择的发行版、版本和体系结构列表。一个很好的例子是“ubuntu”、“bionic”（18.04lts）和“i386”。一会儿之后你的容器将会创建完毕，你可以输入以下命令启动它：

```
lxc-start -n my-container -d
```

然后你可以使用以下任一方法确认其状态

```
lxc-info -n my-container
lxc-ls -f
```

在容器中安装 shell：

```
lxc-attach -n my-container
```

关闭容器：

```
lxc-stop -n my-container
```

最后用以下方法移除：

```
lxc-destroy -n my-container
```

-------------------------------

<h3 id="shadow">shadow解析</h3>

shadow 是 passwd 的影子文件，与/etc/passwd文件不同，/etc/shadow文件是只有系统管理员才有权利进行查看和修改的文件。

shadow 是一个文件，它包含系统账户的密码信息和可选的年龄信息。如果没有维护好密码安全，此文件绝对不能让普通用户可读。此文件的每行包括 9 个字段，使用半角冒号 (“:”) 分隔，顺序如下：


**登录名**

必须是有效的账户名，且已经存在于系统中。

**加密了的密码**

这里保存的是真正加密的密码。目前 Linux 的密码采用的是 SHA512 散列加密算法，原来采用的是 MD5 或 DES 加密算法。SHA512 散列加密算法的加密等级更高，也更加安全。

注意，这串密码产生的乱码不能手工修改，如果手工修改，系统将无法识别密码，导致密码失效。很多软件透过这个功能，在密码串前加上 "!"、"*" 或 "x" 使密码暂时失效。

所有伪用户的密码都是 "!!" 或 "*"，代表没有密码是不能登录的。当然，新创建的用户如果不设定密码，那么它的密码项也是 "!!"，代表这个用户没有密码，不能登录。

**最后一次更改密码的日期**

此字段表示最后一次修改密码的时间，可是，为什么 root 用户显示的是 15775 呢？

这是因为，Linux 计算日期的时间是以  1970 年 1 月 1 日作为 1 不断累加得到的时间，到 1971 年 1 月 1 日，则为 366 天。这里显示 15775 天，也就是说，此 root 账号在 1970 年 1 月 1 日之后的第 15775 天修改的 root 用户密码。

那么，到底 15775 代表的是哪一天呢？可以使用如下命令进行换算：

[root@localhost ~]# date -d "1970-01-01 15775 days"

2013年03月11日 星期一 00:00:00 CST

可以看到，通过以上命令，即可将其换算为我们习惯的系统日期。

**最小修改时间间隔**

最小修改间隔时间，也就是说，该字段规定了从第 3 字段（最后一次修改密码的日期）起，多长时间之内不能修改密码。如果是 0，则密码可以随时修改；如果是 10，则代表密码修改后 10 天之内不能再次修改密码。

此字段是为了针对某些人频繁更改账户密码而设计的。

**最大密码年龄**

经常变更密码是个好习惯，为了强制要求用户变更密码，这个字段可以指定距离第 3 字段（最后一次更改密码）多长时间内需要再次变更密码，否则该账户密码进行过期阶段。

该字段的默认值为 99999，也就是 273 年，可认为是永久生效。如果改为 90，则表示密码被修改 90 天之后必须再次修改，否则该用户即将过期。管理服务器时，通过这个字段强制用户定期修改密码。

**密码警告时间段**

与第 5 字段相比较，当账户密码有效期快到时，系统会发出警告信息给此账户，提醒用户 "再过 n 天你的密码就要过期了，请尽快重新设置你的密码！"。

该字段的默认值是 7，也就是说，距离密码有效期的第 7 天开始，每次登录系统都会向该账户发出 "修改密码" 的警告信息。

**密码禁用期**

也称为“口令失效日”，简单理解就是，在密码过期后，用户如果还是没有修改密码，则在此字段规定的宽限天数内，用户还是可以登录系统的；如果过了宽限天数，系统将不再让此账户登陆，也不会提示账户过期，是完全禁用。

比如说，此字段规定的宽限天数是 10，则代表密码过期 10 天后失效；如果是 0，则代表密码过期后立即失效；如果是 -1，则代表密码永远不会失效。

**账户过期日期**

同第 3 个字段一样，使用自  1970 年 1 月 1 日以来的总天数作为账户的失效时间。该字段表示，账号在此字段规定的时间之外，不论你的密码是否过期，都将无法使用！

该字段通常被使用在具有收费服务的系统中。

空字段表示账户永不过期。

应该避免使用 0，因为它既能理解成永不过期也能理解成在1970年1月1日过期。

**保留字段**

此字段保留作将来使用。

<h3 id="libpam-cgfs">libpam-cgfs</h3>

<a href="https://packages.ubuntu.com/search?keywords=libpam-cgfs">libpam-cgfs</a> 是 Cgroup 当中的 PAM 模块。

容器是系统内部的隔离区域，它们有自己的文件系统、网络、PID、IPC、CPU 和内存分配的名称空间，并且可以使用 Linux 内核中包含的控制组(cgroup)和名称空间(namespace)功能来创建。这提供了一个可插入的身份验证模块（PAM），为登录用户提供一组 cgroup，供他们管理。例如，这允许使用非特权容器和使用 cgroup 进程跟踪进行会话管理。（详情请看<a href="https://github.com/laneston/Note/blob/master/LXC_LearningNote.md">LXC容器概述</a>）

----------------------------

## 作为root用户创建非权限容器

要运行系统范围内的非特权容器（即由root用户启动的非特权容器），只需执行上述步骤的一个子集。

具体来说，你需要手动将 uid 和 gid 范围分配给 /etc/subuid 和 /etc/subgid 的根目录。然后使用与上面类似的 lxc.idmap 条目在 /etc/lxc/default.conf 中设置该范围。

Root 不需要网络设备配额，并且使用全局配置文件，因此其他步骤不适用。此时，作为根用户创建的任何容器都将以非特权运行。

## 创建权限容器

权限容器是 root 用户创建的工作在 root 模式下的容器。

根据Linux发行版的不同，通过一些功能删除、apparmor 配置文件、selinux 上下文或 seccomp 策略，它们可能受到保护。但最终，进程仍以root身份运行，因此你永远不应将对特权容器中 root 的访问权限授予不受信任的一方。

**注：**

*seccomp 是内核提供的一种SYSCALL过滤机制，它基于 BPF 过滤方法，通过写入BPF过滤器代码来达到过滤的目的。BPF 规则语言原生是为了过滤网络包，情景比较复杂。针对 SYSCALL 场景，语法比较固定，可以自行撰写，也可以基于 Libseccomp 库提供的 API 来编写。*

*因为程序在 fork/clone 或 execve 时，BPF filter 会从父进程继承到子进程，所以如果想控制第三方的程序调用SYSCALL，只需要在 fork/clone 或者 execve 时，传入合适的 sock_filter 即可。*

如果您仍然需要创建特权容器，这很简单。只要不做上述任何配置，LXC就会创建特权容器。

```
sudo lxc-create -t download -n privileged-container
```

这将会使用一个下载模板镜像来创建一个新的 "privileged-container" 特权容器在你的系统中。
