# dockerfile 常用命令

1. FROM：指定基础镜像，必须为第一个命令
2. MAINTAINER: 维护者信息
3. RUN：构建镜像时执行的命令
4. ADD：将本地文件添加到容器中，tar类型文件会自动解压(网络压缩资源不会被解压)，可以访问网络资源，类似wget
5. COPY：功能类似ADD，但是是不会自动解压文件，也不能访问网络资源
6. CMD：构建容器后调用，也就是在容器启动时才进行调用。
7. ENTRYPOINT：配置容器，使其可执行化。配合CMD可省去"application"，只使用参数。
8. LABEL：用于为镜像添加元数据
9. ENV：设置环境变量
10. EXPOSE：指定于外界交互的端口
11. VOLUME：用于指定持久化目录
12. WORKDIR：工作目录，类似于cd命令
13. USER : 指定运行容器时的用户名或 UID，后续的 RUN 也会使用指定用户。使用USER指定用户时，可以使用用户名、UID或GID，或是两者的组合。当服务不需要管理员权限时，可以通过该命令指定运行用户。并且可以在之前创建所需要的用户
14. ARG：用于指定传递给构建运行时的变量
15. ONBUILD：用于设置镜像触发器


## 基础镜像制作

可以在 [dockerhub](https://hub.docker.com/) 网站上下载所需的基础镜像，然后启动容器，进入容器，在容器中安装所需的工具，再使用
docker commit 打包成新的镜像。

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-dockerfile/dockerImg.png"></div>

上图中，none 是打包时构造的中间镜像；edge-harbor.iotbi.cc:9443/kubetest/podtest:v1是制作的目标镜像；gcc:cmake是基于gcc:latest制作的基础镜像，用于提供编译环境；Ubuntu:latest是用作打包的基础镜像。


# 文件编写

## cmake文件

CMakeLists.txt
```
cmake_minimum_required (VERSION 3.0)
project(podtest VERSION 2.1.0)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/*.c")
include_directories("${PROJECT_BINARY_DIR}")

add_executable(${PROJECT_NAME} ${SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
```

## c文件
podtest.c
```
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    unsigned int counter = 0;

    while (true)
    {
        if (counter > 65535)
        {
            counter = 0;
        }

        printf("podtest counter: %d\n", counter);
        counter++;

        sleep(30);
    }

    return 0;
}

```

## dockerfile文件

Dockerfile
```
FROM gcc:cmake as gccbuilder

COPY src/ /home/src
WORKDIR /home/src/build
RUN cmake .. && make

FROM ubuntu:latest

COPY --from=gccbuilder /home/src/build/podtest /root/
WORKDIR /root
RUN chmod 777 -R *
ENTRYPOINT ["/root/podtest"]
```

# 编译过程


## 登录私有harbor

这一步非必须，主要用作镜像的拉取或推送前的准备。

```
docker login edge-harbor.iotbi.cc:9443 --username=admin --password="password"
```

## 制作 docker 镜像

当前是非跨平台制作，如果是编译其他架构平台（如arm64）需使用 docker buildx build 命令制作，并使用
--platform=linux/arm64 指定相关平台。

```
docker build -t edge-harbor.iotbi.cc:9443/kubetest/podtest:v1 .
```


## 推送harbor

这一步非必须，目的是为了让其他设备可以通过网络加载镜像。

```
docker push edge-harbor.iotbi.cc:9443/kubetest/podtest:v1
```