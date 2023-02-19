平台: cortex-a53
系统: openWrt

## openssl 编译

paho.mqtt 依赖 OpenSSL 库文件，所以需要先交叉编译 openssl 文件。

```
./config no-asm shared --prefix=/home/openssl-OpenSSL_1_1_1o/__install --cross-compile-prefix=/home/openwrt-toolchain-mediatek-mt7622_gcc-11.3.0_musl/toolchain-aarch64_cortex-a53_gcc-11.3.0_musl/bin/aarch64-openwrt-linux- '-Wl,-rpath,/etc/libs'
```


```
make
make install
```


## paho.mqtt 编译


```
cmake .. 	
	-DCMAKE_TOOLCHAIN_FILE=../mtk_setup.cmake \
	-DPAHO_HIGH_PERFORMANCE=FALSE \
	-DPAHO_WITH_SSL=TRUE \
	-DPAHO_BUILD_DOCUMENTATION=FALSE \
	-DPAHO_BUILD_SAMPLES=TRUE \
	-DMQTT_TEST_BROKER=tcp://localhost:1883 \
	-DMQTT_TEST_PROXY=tcp://localhost:1883 \
	-DMQTT_SSL_HOSTNAME=localhost \
	-DPAHO_BUILD_DEB_PACKAGE=FALSE \
	-DCMAKE_INSTALL_PREFIX=/home/paho.mqtt.c-1.3.10/__install/
```

交叉编译时使用 CMAKE_TOOLCHAIN_FILE 指定编译工具链, 将工具链文件放在 cmakelist.txt 文件所在的目录下即可， mtk_setup.cmake 文件内容如下:
```
# set the tagrt system
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# set the toolchain path
# set(TOOL_CHAIN_DIR /home/openwrt-toolchain-mediatek-mt7622_gcc-8.3.0/toolchain-aarch64_cortex-a53_gcc-8.3.0_musl)
# set(TOOL_CHAIN_DIR /home/kingc/openwrt-gcc-8.3.0)
set(TOOL_CHAIN_DIR /home/toolchain-mediatek-mt7622_gcc-8.3.0_musl/toolchain-aarch64_cortex-a53_gcc-8.3.0_musl/)


# set the toolchain lib & include files
set(TOOL_CHAIN_INCLUDE ${TOOL_CHAIN_DIR}/include)
set(TOOL_CHAIN_LIB ${TOOL_CHAIN_DIR}/lib)


# set the compiler
set(CMAKE_C_COMPILER "${TOOL_CHAIN_DIR}/bin/aarch64-openwrt-linux-gcc")
# set(CMAKE_CXX_COMPILER "${TOOL_CHAIN_DIR}/bin/aarch64-openwrt-linux-g++")

# set the cmake find root path
set(CMAKE_FIND_ROOT_PATH ${TOOL_CHAIN_DIR})

# search for programs in the build host directories (not necessary)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# 只在指定目录下查找库文件,只在指定目录下查找头文件,只在指定目录下查找依赖包
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_INCLUDE_PATH ${TOOL_CHAIN_INCLUDE})
set(CMAKE_LIBRARY_PATH ${TOOL_CHAIN_LIB})
```

执行以下命令即可完成编译与安装至指定目录下。

```
make
make install
```