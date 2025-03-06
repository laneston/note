


用以下命令查看Go支持的目标平台和架构:
```
go tool dist list
```

用以下命令查看当前环境变量设置:
```
go env
```

# 交叉编译

**环境：** MTK7622

```
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 AR=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-ar CC=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-gcc CXX=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-g++ go build hello_world.go
```



```
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 AR=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-ar CC=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-gcc CXX=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-g++ make all WHAT=cloudcore
```

```
export STAGING_DIR=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin:$STAGING_DIR
```
```
CGO_ENABLED=1 GOOS=linux GOARCH=arm64 AR=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-ar CC=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-gcc CXX=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-g++ make all WHAT=edgecore
```


```
CGO_ENABLED=1 GOOS=linux GOARCH=arm64 AR=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-ar CC=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-gcc CXX=/home/toolchain-aarch64_cortex-a53_gcc-11.2.0_musl/bin/aarch64-openwrt-linux-g++ make all WHAT=keadm
```


```
mkdir /etc/kubeedge/config/ && cd /etc/kubeedge/config/
```


./keadm gettoken --kube-config=$HOME/.kube/config


24eb6189eba66f2854a7bc299d864ff4123946035d38b6204a58b09256ff9191.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2NzM5MTUyNjB9.8JFf0SPx0s6d8iprZ1EZdNNTdnGkQqWWvXVLrEYMHlA