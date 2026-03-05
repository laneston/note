mosquitto交叉编译和依赖性编译过程如下：

# cJSON

## 下载 cJSON (以 1.7.16 为例，版本可自选)
wget https://codeload.github.com/DaveGamble/cJSON/tar.gz/refs/tags/v1.7.16
tar -xzf v1.7.16 && cd cJSON-1.7.16/


## 创建构建目录
mkdir build && cd build

## 配置 CMake 进行交叉编译
cmake .. \
    -DCMAKE_C_COMPILER=/home/lanceli/arm-buildroot-linux-gnueabihf_sdk-buildroot/bin/arm-buildroot-linux-gnueabihf-gcc \
    -DCMAKE_INSTALL_PREFIX=~/cjson-cross/install \
    -DBUILD_SHARED_AND_STATIC_LIBS=ON \
    -DENABLE_TARGET_EXPORT=OFF \
    -DCMAKE_BUILD_TYPE=Release

## 编译并安装
make
make install
cd ../..



# c-ares

## 下载 c-ares (以 1.34.6 为例)
wget https://codeload.github.com/c-ares/c-ares/tar.gz/refs/tags/v1.34.6
tar -xzf v1.34.6
cd c-ares-1.34.6/


## 创建构建目录
mkdir build && cd build


## 配置 CMake 进行交叉编译
cmake .. \
    -DCMAKE_C_COMPILER=/home/lanceli/arm-buildroot-linux-gnueabihf_sdk-buildroot/bin/arm-buildroot-linux-gnueabihf-gcc \
    -DCMAKE_INSTALL_PREFIX=~/c-ares/install \
    -DBUILD_SHARED_AND_STATIC_LIBS=ON


## 编译并安装
make
make install
cd ..

# sqlite
## 下载 sqlite
wget https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz

## 解压
tar -xzvf sqlite-autoconf-3450200.tar.gz && cd sqlite-autoconf-3450200/

## 根据你的工具链实际路径设置
export TOOLCHAIN_PATH=/home/lanceli/arm-buildroot-linux-gnueabihf_sdk-buildroot/bin && export PATH=$PATH:$TOOLCHAIN_PATH

./configure \
    --host=arm-buildroot-linux-gnueabihf \
    --prefix=$PWD/install \
    CC=arm-buildroot-linux-gnueabihf-gcc \
    CXX=arm-buildroot-linux-gnueabihf-g++ \
    CFLAGS="-O3 -march=armv7-a"


## 编译并安装
make
make install   # 如需安装到指定目录



# GTest
## 下载 GTest 源码（建议使用较新且稳定的 release 版本，如 v1.14.0）
## 注意：针对老旧的 ARM 工具链，可能需要选用更早的版本（如 v1.8.1）以避免 C++11 特性不支持的问题 [citation:2]
wget https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
tar -xzf v1.14.0.tar.gz && cd googletest-1.14.0/


## 在源码根目录下创建 build 目录并进入
mkdir build && cd build

## 配置 CMake 进行交叉编译
cmake .. \
    -DCMAKE_C_COMPILER=/home/lanceli/arm-buildroot-linux-gnueabihf_sdk-buildroot/bin/arm-buildroot-linux-gnueabihf-gcc \
    -DCMAKE_CXX_COMPILER=/home/lanceli/arm-buildroot-linux-gnueabihf_sdk-buildroot/bin/arm-buildroot-linux-gnueabihf-g++ \
    -DCMAKE_INSTALL_PREFIX=~/gtest-cross/install \
    -DCMAKE_BUILD_TYPE=Release

## 编译并安装
make
make install   # 如需安装到指定目录




# CUnit

## 下载 CUnit 源码（以 2.1-3 版本为例，这是较新的稳定版）
wget https://sourceforge.net/projects/cunit/files/CUnit/2.1-3/CUnit-2.1-3.tar.bz2/download -O CUnit-2.1-3.tar.bz2
tar -xjf CUnit-2.1-3.tar.bz2 && cd CUnit-2.1-3/

## 定义工具链前缀和路径
export CROSS_COMPILE=/home/lanceli/arm-buildroot-linux-gnueabihf_sdk-buildroot/bin/arm-buildroot-linux-gnueabihf- && \
export CC=${CROSS_COMPILE}gcc && \
export AR=${CROSS_COMPILE}ar && \
export RANLIB=${CROSS_COMPILE}ranlib && \
export LD=${CROSS_COMPILE}ld && \
export AS=${CROSS_COMPILE}as && \
export CXX=${CROSS_COMPILE}g++

## 安装 autotools（如果尚未安装）
sudo apt update
sudo apt install autoconf automake libtool

## 在源码根目录下运行 autoreconf
autoreconf -i -f

这会生成 configure、missing、install-sh 等文件。之后再次运行 ./configure 进行交叉编译配置。


./configure \
    --host=arm-linux-gnueabihf \
    --prefix=$HOME/cunit-cross/install \
    --disable-shared \
    --enable-static=yes \
    CC=$CC \
    AR=$AR \
    RANLIB=$RANLIB


## 编译并安装
make
make install   # 如需安装到指定目录





# mosquitto


# 下载 mosquitto (以 2.1.2 为例)
wget https://codeload.github.com/eclipse-mosquitto/mosquitto/tar.gz/refs/tags/v2.1.2
tar -xzf v2.1.2 && cd mosquitto-2.1.2/


## 创建构建目录
mkdir build && cd build

## 配置 CMake 进行交叉编译

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=~/aarch32-toolchain.cmake \
    -DCMAKE_INSTALL_PREFIX=~/mosquitto/install \
    -DWITH_DOCS=no \
    -DWITH_SQLITE=no

## 编译并安装
make
make install   # 如需安装到指定目录

## 配置

在 `aarch32` 环境（如嵌入式ARM设备）的 Linux 系统中，Mosquitto 的配置方式与在其他架构（如 x86）上完全一致。配置文件同样位于 `/etc/mosquitto/mosquitto.conf`。

要实现“允许HOST下的客户端无需用户名密码连接”，本质上就是配置**允许匿名访问**，并确保服务**监听所有网络接口**（而不仅仅是本地回环）。以下是具体配置步骤和参数说明。

### 1. 定位配置文件
Mosquitto 的主配置文件路径为：
```bash
/etc/mosquitto/mosquitto.conf
```
在 `aarch32` 的 Debian/Ubuntu 或 OpenWrt 等嵌入式系统中，该路径通常保持不变 。

### 2. 配置参数详解
使用文本编辑器（如 `vi` 或 `nano`）编辑配置文件：
```bash
sudo vi /etc/mosquitto/mosquitto.conf
```
为了满足你的需求，核心需要关注以下两个参数：

*   **`allow_anonymous`**：此参数控制是否允许不提供用户名密码的客户端连接。
    *   **配置值**：设置为 `true` 即允许匿名连接 。
    *   **注意**：如果此参数被设置为 `false` 或指向了 `password_file`，则必须提供用户名密码才能连接 。

*   **`listener`**：此参数定义 Mosquitto 监听的端口和 IP 地址。**这是实现远程连接的关键**。
    *   **背景**：从 Mosquitto 2.0 版本开始，为了增强安全性，如果只使用命令行启动或没有配置 `listener`，默认监听器只会绑定到 `localhost`（127.0.0.1），这意味着只能从本机连接，无法从外部 HOST 访问 。
    *   **配置值**：需要显式定义一个监听器。最简单的配置是 `listener 1883`，这会使 Mosquitto 监听所有网络接口（`0.0.0.0`）的 1883 端口 。

### 3. 最小配置示例
为了实现你的目标，配置文件中最少需要包含以下两行。你可以将它们添加到文件末尾，或确保文件中存在且未被注释掉：

```
# 监听所有网络接口的 1883 端口，允许外部 HOST 连接
listener 1883

# 允许客户端匿名登录（无需用户名密码）
allow_anonymous true
```
将此配置与默认配置结合，即可满足需求 。

### 4. 重启服务应用配置
修改配置文件后，需要重启 Mosquitto 服务使配置生效。在 `aarch32` 的 systemd 系统中，通常使用以下命令：
```bash
sudo systemctl restart mosquitto
```
重启后，可以通过以下命令检查服务状态和监听端口，确认 Mosquitto 是否已正确监听 `0.0.0.0:1883`：
```bash
sudo systemctl status mosquitto
sudo netstat -tlnp | grep 1883
```

完成上述步骤后，你 HOST 端的客户端即可使用 `aarch32` 设备的 IP 地址和端口 `1883` 进行无密码连接。