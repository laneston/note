# docker 日志查看

**时间区间日志：**

```
docker logs --since='2022-01-14T00:58:00' --until='2022-01-14T01:00:00' ContainerID
```

**查看全部日志：**

```
docker logs -f ContainerID
```

**查看指定行数日志：**

```
docker logs --tail=100 ContainerID
```

# 容器与宿主机时间戳不一致

在docker容器和系统时间不一致是因为docker容器的原生时区为0时区，而宿主机时间是+8个时区，需修改原镜像时区。


