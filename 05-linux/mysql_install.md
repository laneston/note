# 编译准备

- 主机编译平台：ubuntu20.04 (Windows 子系统)
- 交叉编译链：aarch64-none-linux-gnu-
- mysql版本：mysql-5.7.32
- mysql运行环境：openwrt

mysql交叉编译的主要流程是：主机编译 mysql，交叉编译 boost 库，交叉编译 ncurse 库，交叉编译 openssl 库，最后交叉编译 mysql 库。值得注意的是，mysql 对依赖库的版本有特殊指定，请尽量按照本说明对应的版本进行操作，如果依赖库没有达到 mysql 指定的版本新度，会出现配置不通过。

以下我们就开始按编译顺序一步步操作吧。

## mysql的主机编译

之所以第一步要主机编译 mysql，第一是为了熟悉编译的流程，但最重要的是，我们之后交叉编译需要用到主机编译时生成的脚本。

编译过程在此说明。

解压 mysql-5.7.32 压缩包，进入文件夹第一层目录，输入以下命令：

```
cmake . -DDOWNLOAD_BOOST=1 -DWITH_BOOST=/home/mysqlCompile/mysql-5.7.32/boost -DCMAKE_INSTALL_PREFIX=/home/mysqlCompile/mysql-5.7.32/__install
```

这个命令是配置命令，目的是生成编译的 Makefile 文件。这个命令的意思是：在当前目录生成 Makefile 文件，并在 /home/Mysql_Complie/mysql-8.0.22/boost 路径下 download boost 库。因为编译 mysql 库时需要依赖 boost。除此之外，主机也需要安装 openssl-dev 库，与 ncurse 库。如果是 ubuntu 环境，可在编译之前输入以下命令进行安装：

```
apt-get install openssl
apt-get install libssl-dev
```

而 -DCMAKE_INSTALL_PREFIX=/home/Mysql_Complie/mysql-8.0.22/__install 一句是指定安装路径。

如无意外，输入以上配置命令后我们已经生成了 Makefile 文件了，接着我们分别输入以下命令就可以进行编译与安装:

```
make
make install
```

这样，我们就能在之前定义的安装路径下找到编译好的 mysql 文件。

## 编译boost库

源码 <a href="https://www.boost.org/users/download/">在此下载</a>

本次编译的版本为：boost_1_59_0

需要注意的是，不同版本的 mysql 需要的 boost 的版本可能会有不同，编译时需要注意错误提示，并交叉编译对应的版本。

解压之后进入文件夹内，输入以下命令进行编译配置：

```
./bootstrap.sh  --prefix=/home/mysqlCompile/boost_1_59_0/__install
```

相应的文件路径可以根据自己本地的实际文件路径进行修改。

此次配置会生成名为 b2 的执行文件，配置结束后需在 project-config.jam 文件中修改交叉编译链的声明：

```
if ! gcc in [ feature.values <toolset> ]
{
    using gcc : : /home/lance/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc ;
}
```

**值得注意的是，冒号间与分号前空格的存在，那是必须的。**

接着我们依次输入以下命令即可进行代码的编译与库的安装：

```
./b2

./b2 install
```

这个也可以在配置过程中设置参数使工程自动下载：

```
-DENABLE_DOWNLOADS=1 \
-DDOWNLOAD_BOOST=1 \
-DWITH_BOOST=/home/mysqlCompile/mysql-8.0.22/boost
```

但是稳妥起见，且为了之后的库移植，最好事先交叉编译好。

## 编译ncurse库

解压之后进入文件夹内，输入以下命令配置编辑方式与安装路径：

```
./configure --prefix=/home/mysqlCompile/ncurses-6.2/__install --host=aarch64-none-linux-gnu  CC=aarch64-none-linux-gnu-gcc --with-shared  --without-progs
```

编译与安装：

```
make
make install
```

分别输入以上命令即可对 ncurse 库进行编译与安装。

## 编译openssl库

执行以下命令，分别对 openssl 库进行配置编译与安装。

```
./config --prefix=/home/mysqlCompile/openssl-OpenSSL_1_1_1g/__install --cross-compile-prefix=aarch64-none-linux-gnu- no-asm shared
make
make install
```

以上相对路径可根据自己的本地路径进行修改。

## mysql的交叉编译

这是本篇文章的核心部分，mysql 交叉编译的过程主要是通过 cmake 生成相应的配置文件与 Makefile，然后再执行 Makefile 脚本文件生成相应的目标文件。在用 cmake 生成 Makefile 文件之间，我们需要对 mysql 工程进行一些修改。

### CMakeLists.txt文件的修改

打开一级目录下的 CMakeLists.txt 文件，将以下配置信息复制到文件首部，并保存文件，当然，其中的文件路径需要根据自己的编译环境进行修改。

```
# this is required
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_CROSSCOMPILING TRUE)

# specify the cross compiler
SET(CMAKE_C_COMPILER /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)

# where is the target environment 
SET(CMAKE_FIND_ROOT_PATH  /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu)

# search for programs in the build host directories (not necessary)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# configure Boost
SET(BOOST_ROOT /home/mysqlCompile/boost_1_59_0/__install)
SET(BOOST_INCLUDE_DIR /home/mysqlCompile/boost_1_59_0/__install/include)
SET(BOOST_LIBRARY_DIR /home/mysqlCompile/boost_1_59_0/__install/lib)

# openssl configuration
SET(OPENSSL_INCLUDE_DIR /usr/local/opt/openssl/include)
SET(OPENSSL_LIBRARY /usr/local/opt/openssl/lib/libssl.so)
SET(CRYPTO_LIBRARY /usr/local/opt/openssl/lib/libcrypto.so)

SET(CMAKE_CXX_LINK_FLAGS "-L/usr/local/opt/openssl/lib -lssl -lcrypto")
```

这里分别指定了：

1. 编译的目标环境；
2. 交叉编译链工具的路径信息；
3. 交叉编译的环境信息；
4. cmake 寻找链接文件的规则；
5. boost 库的路径信息；
6. openssl库路径信息。

修改完 CMakeLists.txt 文件后，我们还需要修改 libevent.cmake 文件。

## 修改 libevent.cmake 文件

生成配置文件的过程中会有找不到 libevent-2.1.11-stable 版本信息的警告。

这个问题是由于生成的目标文件是运行在目标平台上的ARM文件，由于平台的不同，即便 TRY_RUN() 函数执行这个目标文件，文件也是无法在主机平台上运行的。而这和目标文件的动作目的是获取 libevent 库的版本信息。为了使配置成功执行，我们需要做的是：让 libevent 库的版本信息成功写入到配置文件当中。以下是需要修改的部分：

```
MACRO(FIND_LIBEVENT_VERSION)

  SET(LIBEVENT_VERSION "2.1.11-stable")
  SET(COMPILE_TEST_RESULT TRUE)
  SET(RUN_OUTPUT "2.1.11-stable")

  # MESSAGE(STATUS "TRY_EVENT TEST_RUN_RESULT is ${TEST_RUN_RESULT}")
  # MESSAGE(STATUS "TRY_EVENT COMPILE_TEST_RESULT is ${COMPILE_TEST_RESULT}")
  # MESSAGE(STATUS "TRY_EVENT COMPILE_OUTPUT_VARIABLE is ${OUTPUT}")
  # MESSAGE(STATUS "TRY_EVENT RUN_OUTPUT_VARIABLE is ${RUN_OUTPUT}")

  IF(COMPILE_TEST_RESULT)
    SET(LIBEVENT_VERSION_STRING "${RUN_OUTPUT}")
    STRING(REGEX REPLACE
      "([.-0-9]+).*" "\\1" LIBEVENT_VERSION "${LIBEVENT_VERSION_STRING}")
    MESSAGE(STATUS "LIBEVENT_VERSION_STRING ${LIBEVENT_VERSION_STRING}")
    MESSAGE(STATUS "LIBEVENT_VERSION (${WITH_LIBEVENT}) ${LIBEVENT_VERSION}")
  ELSE()
    MESSAGE(WARNING "Could not determine LIBEVENT_VERSION")
  ENDIF()
ENDMACRO()
```

上面 cmake 命令修改的部分是将版本信息直接定义到 LIBEVENT_VERSION 变量中，跳过了代码编译与运行的步骤。

修改完毕后执行一次 cmake 命令：

```
cmake . -DENABLE_DOWNLOADS=1 -DWITH_BOOST= /home/mysqlCompile/boost_1_59_0/__install -DCMAKE_INSTALL_PREFIX=/home/mysqlCompile/mysql-5.7.32/__install -DCURSES_INCLUDE_PATH=/home/mysqlCompile/ncurses-6.2/__install/include -DCURSES_LIBRARY=/home/mysqlCompile/ncurses-6.2/__install/lib/libncurses.so -DSTACK_DIRECTION=1 -DWITH_LIBEVENT="bundled"
```

如果不能正常生成 Makefile 文件并伴有以下错误信息 ：

```
CMake Error: TRY_RUN() invoked in cross-compiling mode, please set the following cache variables appropriately: 
```

可再执行一次上一次的配置命令，如无意外，我们就能获得 Makefile 文件了，在配置的过程当中，会有文件需要下载，但服务器在国外，如果有梯子，请搭上。

此时可以输入 make 命令进行编译，不出一会儿，这时我们就遇到了编译过程中的第一个错误。

# 开始编译

## /bin/sh: 1: comp_err: not found

这个错误从字面上理解，就是这个脚本没有发现。解决办法就是把我们之前本机编译得到的 comp_err 文件移动到编译环境中的 bin 文件夹中：

```
cp /home/mysql-5.7.32/extra/comp_err /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/
touch /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/comp_err
```

继续执行 make 命令进行编译，我们会遇到第二个错误。

## /bin/sh: 1: ./libmysql_api_test: Exec format error

解决办法是将本机编译得到的 libmysql_api_test 文件，移动到交叉编译对应的文件夹中：
```
cp /home/mysql-5.7.32/libmysql/libmysql_api_test /home/mysqlCompile/mysql-5.7.32/libmysql/
```

继续执行 make 命令进行编译，我们会遇到第三个错误。

## error "Unsupported platform"

```
In file included from /home/mysqlCompile/mysql-5.7.32/storage/innobase/include/os0atomic.h:375,
                 from /home/mysqlCompile/mysql-5.7.32/storage/innobase/include/ut0ut.h:47,
                 from /home/mysqlCompile/mysql-5.7.32/storage/innobase/include/univ.i:591,
                 from /home/mysqlCompile/mysql-5.7.32/storage/innobase/include/ha_prototypes.h:40,
                 from /home/mysqlCompile/mysql-5.7.32/storage/innobase/api/api0api.cc:35:
/home/mysqlCompile/mysql-5.7.32/storage/innobase/include/os0atomic.ic:230:2: error: #error "Unsupported platform"
  230 | #error "Unsupported platform"
```

这个问题如提示所示，是平台不支持，原因是宏定义的问题。


os0atomic.ic 中有 HAVE_IB_GCC_ATOMIC_COMPARE_EXCHANGE 与 IB_STRONG_MEMORY_MODEL 这两个宏定义。


在 os0atomic.h 的 60 行附近， 从以上的内容可以看出，只有定义了

```
 __i386__ || __x86_64__ || _M_IX86 || _M_X64 || __WIN__ 
```

才能定义 IB_STRONG_MEMORY_MODEL，但是我们是交叉编译 mysql，平台是 arm，明显上面的内容没有定义，所以在交叉编译的时候就没有定义，导致 os0atomic.ic 中的内容没有编译。

修改办法如下：

```
#if defined __i386__ || defined __x86_64__ || defined _M_IX86 \
    || defined _M_X64 || defined __WIN__

#define IB_STRONG_MEMORY_MODEL

#else

#define HAVE_ATOMIC_BUILTINS

#endif /* __i386__ || __x86_64__ || _M_IX86 || _M_X64 || __WIN__ */
```

HAVE_ATOMIC_BUILTINS 这个宏不是随便定义的，可在文章  <a href="https://developer.aliyun.com/article/51094">MariaDB · 社区动态 · MariaDB on Power8</a> 中了解到。

但这样并不能完全解决问题，在 os0atomic.h 文件中找到 os_compare_and_swap_thread_id() 这个函数的定义，而在这个函数的前面有编译条件如下：

```
# ifdef HAVE_IB_ATOMIC_PTHREAD_T_GCC
#if defined(HAVE_GCC_SYNC_BUILTINS)
#  define os_compare_and_swap_thread_id(ptr, old_val, new_val) \
	os_compare_and_swap(ptr, old_val, new_val)
#else
UNIV_INLINE
bool
os_compare_and_swap_thread_id(volatile os_thread_id_t* ptr, os_thread_id_t old_val, os_thread_id_t new_val)
{
  return __atomic_compare_exchange_n(ptr, &old_val, new_val, 0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
#endif /* HAVE_GCC_SYNC_BUILTINS */
```

但是交叉编译工具GCC中没有这两个宏，所以运行不了，解决方法是改为如下：

```
# ifdef HAVE_ATOMIC_BUILTINS
#if defined(HAVE_ATOMIC_BUILTINS)
#  define os_compare_and_swap_thread_id(ptr, old_val, new_val) \
	os_compare_and_swap(ptr, old_val, new_val)
#else
UNIV_INLINE
bool
os_compare_and_swap_thread_id(volatile os_thread_id_t* ptr, os_thread_id_t old_val, os_thread_id_t new_val)
{
  return __atomic_compare_exchange_n(ptr, &old_val, new_val, 0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
#endif /* HAVE_GCC_SYNC_BUILTINS */
```

继续执行 make 命令进行编译，我们会遇到第四个错误。

## No rule to make target 'scripts/comp_sql'

```
make[2]: *** No rule to make target 'scripts/comp_sql', needed by 'scripts/sql_commands_sys_schema.h'.  Stop.
```

解决办法如下：

```
cp /home/mysql-5.7.32/scripts/comp_sql /home/mysqlCompile/mysql-5.7.32/scripts/
touch  /home/mysqlCompile/mysql-5.7.32/scripts/comp_sql
```

继续执行 make 命令进行编译，我们会遇到第五个错误。

## /bin/sh: 1: comp_sql: not found

解决办法如下：

```
cp /home/mysql-5.7.32/scripts/comp_sql /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/
touch /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/comp_sql
```

继续执行 make 命令进行编译，我们会遇到第六个错误。

## /bin/sh: 1: gen_lex_hash: not found

解决办法如下：

```
cp /home/mysql-5.7.32/sql/gen_lex_hash /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/
touch /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/gen_lex_hash
```

继续执行 make 命令进行编译，我们会遇到第七个错误。

## /bin/sh: 1: gen_lex_token: not found

解决办法如下：

```
cp /home/mysql-5.7.32/sql/gen_lex_token /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/
touch /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/gen_lex_token
```

继续执行 make 命令进行编译，我们会遇到第八个错误。

## cannot find -ltirpc

```
/home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/../lib/gcc/aarch64-none-linux-gnu/10.2.1/../../../../aarch64-none-linux-gnu/bin/ld: cannot find -ltirpc
collect2: error: ld returned 1 exit status
make[2]: *** [rapid/plugin/group_replication/CMakeFiles/group_replication.dir/build.make:1488: rapid/plugin/group_replication/group_replication.so] Error 1
make[1]: *** [CMakeFiles/Makefile2:5620: rapid/plugin/group_replication/CMakeFiles/group_replication.dir/all] Error 2
make: *** [Makefile:163: all] Error 2
```

这个问题我们需要查看 rapid/plugin/group_replication/CMakeFiles/group_replication.dir/build.make 文件的第 1488 行：

```
cd /home/mysqlCompile/mysql-5.7.32/rapid/plugin/group_replication && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/group_replication.dir/link.txt --verbose=$(VERBOSE)  
```

这里执行了一行命令，而命令的位置在 cmake_link_script CMakeFiles/group_replication.dir/link.txt

这里显示找不到 -ltirpc 这一个指令，我们查看 link.txt 文件，并找出 -ltirpc 将它删除即可。

继续执行 make 命令进行编译，我们会遇到第九个错误。

## /bin/sh: 1: protoc: not found

解决办法如下：

```
cp /home/mysql-5.7.32/extra/protobuf/protoc /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/
root@DESKTOP-PGPFAI6:~# touch /home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/protoc
```

继续执行 make 命令进行编译，我们会遇到第十个错误。

## cannot find -lboost_system -lboost_chrono

这个问题跟之前的 -ltirpc 一样，都是声明后找不到对应的库，这次我们不能讲这两个删除掉，因为编译的过程需要用到这两个库。

```
/home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/../lib/gcc/aarch64-none-linux-gnu/10.2.1/../../../../aarch64-none-linux-gnu/bin/ld: cannot find -lboost_system
/home/lanceli/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/../lib/gcc/aarch64-none-linux-gnu/10.2.1/../../../../aarch64-none-linux-gnu/bin/ld: cannot find -lboost_chrono
collect2: error: ld returned 1 exit status
make[2]: *** [client/dump/CMakeFiles/mysqlpump.dir/build.make:92: client/dump/mysqlpump] Error 1
make[1]: *** [CMakeFiles/Makefile2:12520: client/dump/CMakeFiles/mysqlpump.dir/all] Error 2
make: *** [Makefile:163: all] Error 2
```

根据错误提示，我们打开 client/dump/CMakeFiles/mysqlpump.dir/build.make 文件，定位到第 92 行。

```
cd /home/mysqlCompile/mysql-5.7.32/client/dump && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mysqlpump.dir/link.txt --verbose=$(VERBOSE)
```

打开相同路径下的 link.txt 文件，并将其中 -lboost_system -lboost_chrono 修改为：

```
/home/mysqlCompile/boost_1_59_0/__install/lib/libboost_system.so -ldl /home/mysqlCompile/boost_1_59_0/__install/lib/libboost_chrono.so -ldl
```

继续执行 make 命令进行编译，然后等待代码编译结束。

输入 make install 进行安装，即可在设置的安装路径找到相应的文件。