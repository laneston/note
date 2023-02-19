# 编译环境配置(centos)

- 硬件平台：BPI-R64 (MT7622)
- 主机环境：centos8/WSL


## 软件安装


在正式编译前，我们需要在 Ubuntu 上安装以下工具，以保证编译能够正常执行：


根据 openwrt 编译指南，需要预先安装以下工具：

binutils bzip2 diff find flex gawk gcc-6+ getopt grep install libc-dev libz-dev
make4.1+ perl python3.6+ rsync subversion unzip which



```
yum install binutils
yum install bzip2
yum install flex
yum install gawk
yum install perl
yum install rsync
yum install subversion
yum install unzip
yum install which
yum install gcc-c++
yum install ncurses
yum install ncurses-devel
yum install git
yum install patch
```


# 编译环境配置(ubuntu)

- 硬件平台：MT7622
- 主机环境：Ubuntu20(windows 子系统)
- openwrt 版本：18.06.9

## 软件安装

在正式编译前，我们需要在 Ubuntu 上安装以下工具，以保证编译能够正常执行：

```
/*获取服务器端更新的清单*/
sudo apt-get update
/*安装 git (非必要)*/
sudo apt-get install git
/*安装 g++ 编译工具*/
sudo apt-get install g++
sudo apt-get install gcc
/*GNU C Library: Development Libraries and Header Files (6.0+20160213-1ubuntu1)*/
sudo apt-get install libncurses5-dev
/*GNU C Library: Development Libraries and Header Files (1:1.2.8.dfsg-2ubuntu4.3)*/
sudo apt-get install zlib1g-dev
/*YACC-compatible parser generator*/
sudo apt-get install bison
/*fast lexical analyzer generator*/
sudo apt-get install flex
/*解压工具*/
sudo apt-get install unzip
/*automatic configure script builder*/
sudo apt-get install autoconf
/*GNU awk, a pattern scanning and processing language*/
sudo apt-get install gawk
sudo apt-get install libssl-dev
/*makefile 脚本执行工具*/
sudo apt-get install cmake
sudo apt-get install make
/*GNU Internationalization utilities*/
sudo apt-get install gettext
/*GNU assembler, linker and binary utilities*/
sudo apt-get install binutils
/*Apply a diff file to an original*/
sudo apt-get install patch
/*high-quality block-sorting file compressor - utilities*/
sudo apt-get install bzip2
/*compression library - development*/
sudo apt-get install libz-dev
/* Highly configurable text format for writing documentation [universe]*/
/*其实就是个文本编辑工具，太大，可不安装*/
sudo apt-get install asciidoc
/*Advanced version control system*/
sudo apt-get install subversion
/*Informational list of build-essential packages*/
sudo apt-get install build-essential
/*easy-to-use, scalable distributed version control system [universe]*/
sudo apt-get install mercurial
```

## 下载源码

在 GitHub 上找到 <a href="https://github.com/openwrt/openwrt"> openwrt </a> 源码，选择自己需要的版本进行下载。

# 编译过程

在编译前，将编译文件夹读写权限设置成可读写：

```
sudo chmod 777 -R openwrt-18.06.9
```

## 解压源码包

```
sudo unzip openwrt-18.06.9.zip
```

## 更新软件包

以下操作时为了检测编译环境工具是否齐全，如果存在 failure 字样，可以安装对应工具。

```
sudo ./scripts/feeds update -a
```

如果全部通过，则执行以下操作，安装对应软件包。

```
sudo ./scripts/feeds install -a
```

## 进入定制界面

```
sudo make menuconfig
```

根据所需选取响应配置。


## 下载所需工具包

```
sudo make download
```

## 后台运行

在使用服务器编译时，可以使用以下命令后台运行：

```
nohup cmd &
tail -f nohup.out
```

```
ps -aux | grep "make" 
```

- a : 显示所有程序
- u : 以用户为主的格式来显示
- x : 显示所有程序，不区分终端机



## 开始编译

```
sudo make -j1 FORCE_UNSAFE_CONFIGURE=1 V=s
sudo make FORCE_UNSAFE_CONFIGURE=1 V=s
```

为了增加成功率，首次编译所设置的线程数最好为1： -j1

之后就是漫长的编译过程。


# 错误解决

## 1
```
Checking 'case-sensitive-fs'... failed.
```

原因是文件解压保存在windows的文件夹，因为用的是WSL操作，应该要解压到linux环境的文件夹上，直接拷贝到 /home 目录就可以了。


## 2 

```
cd /home/openwrt/build_dir/host/ninja-1.11.0 && CXX="g++" CXXFLAGS=" -I/home/openwrt/staging_dir/host/include " LDFLAGS="-L/home/openwrt/staging_dir/host/lib " /home/openwrt/staging_dir/host/bin/python3 configure.py --bootstrap
bash: /home/openwrt/staging_dir/host/bin/python3: No such file or directory
```


```
cd /staging_dir/host/bin
sudo rm python*
sudo ln -s /bin/python3.8 python
sudo ln -s /bin/python3.8 python3
```




# 固件烧录

在 /bin目录下找到编译好的固件，如果首次烧录时，可以将 openwrt-mediatek-mt7622-bananapi_bpi-r64-sdcard.img.gz 解压烧录到SD卡中，使用其中的 UBoot 进行相应的操作(将固件刷写到eMMC，或者刷写新的 .itb 镜像)。

# 脚本理解

.its 文件是给 mkimage 用的，用于生成 Image Tree Blob (.itb file)


```
U-Boot firmware supports the booting of images in the Flattened Image Tree (FIT) format.  
The FIT format uses a device tree structure to describe a kernel image, device tree blob, ramdisk, etc. 
This script creates an Image Tree Source (.its file) which can be passed to the'mkimage' utility to generate an Image Tree Blob (.itb file). 
The .itb file can then be booted by U-Boot (or other bootloaders which support FIT images).  
See doc/uImage.FIT/howto.txt in U-Boot source code for additional information on FIT images.
```
