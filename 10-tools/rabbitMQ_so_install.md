# rabbitMQ 动态库交叉编译部署流程

## 交叉编译 openssl

```
./config --prefix=/mnt/d/openssl/openssl-openssl-3.0.2/__install --cross-compile-prefix=aarch64-openwrt-linux- no-asm shared
```
or
```
./config --prefix=/mnt/d/openssl/openssl-OpenSSL_1_1_1o/__install --cross-compile-prefix=aarch64-openwrt-linux- no-asm shared
```


如若遇到 "-m64" 错误警告，去掉 Makefile 对应行处的 -m64 即可。


```
make&make install
```

## 交叉编译 rabbitMQ


将 openssl 库文件放入 /mnt/d/rabbitmq-c-aarch64/openssl_lib 路径下，并删除 /mnt/d/rabbitmq-c-aarch64/openssl_lib/lib 路径下的动态库，只保留静态库文件。



```
cmake -DCMAKE_C_COMPILER=aarch64-openwrt-linux-gcc \
-DCMAKE_CXX_COMPILER=aarch64-openwrt-linux-g++ \
-DOPENSSL_ROOT_DIR=/mnt/d/rabbitmq-c-aarch64/openssl_lib \
-DOPENSSL_CRYPTO_LIBRARY=/mnt/d/rabbitmq-c-aarch64/openssl_lib/lib/libcrypto.a \
-DOPENSSL_LIBRARIES=/mnt/d/rabbitmq-c-aarch64/openssl_lib/lib/libssl.a \
-DOPENSSL_INCLUDE_DIR=/mnt/d/rabbitmq-c-aarch64/openssl_lib/include \
-DCMAKE_INSTALL_PREFIX=/mnt/d/rabbitmq-c-aarch64/__install ..
```

编译安装

```
make&make install
```