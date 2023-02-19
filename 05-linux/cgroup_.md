# docker 与 systemd


systemd和cgroupfs都是CGroup管理器，而systemd是大多数Linux发行版原生的。如果Docker运行时和kubelet的CGroup驱动配置为cgroupfs，则意味着使用了systemd作为init system的系统上有两个不同的CGroup管理器。cgroups(Control Groups) 是 linux 内核提供的一种机制。它可以限制、记录任务组所使用的物理资源。它是内核附加在程序上的hook，使程序运行时对资源的调度触发相应的钩子,达到资源追踪和限制资源使用的目的。docker默认的Cgroup Driver是cgroupfs。cgroupfs是cgroup为给用户提供的操作接口而开发的虚拟文件系统类型，它和sysfs，proc类似，可以向用户展示cgroup的hierarchy，通知kernel用户对cgroup改动。对cgroup的查询和修改只能通过cgroupfs文件系统来进行。

当选择systemd作为Linux发行版的init system时，init process生成并使用一个root控制组(cgroup)，并充当cgroup管理器。Systemd与cgroups紧密集成，并为每个Systemd Unit分配一个cgroup。可以将容器runtime 和kubelet配置为使用cgroupfs。与systemd一起使用cgroupfs意味着将有两个不同的cgroup管理器。


# 为什么要修改为使用systemd

Kubernetes 推荐使用 systemd 来代替 cgroupfs。因为systemd是Kubernetes自带的cgroup管理器, 负责为每个进程分配cgroups,但docker的cgroup driver默认是cgroupfs,这样就同时运行有两个cgroup控制管理器,
当资源有压力的情况时,有可能出现不稳定的情况。

# 更改CGroup管理器

修改docker的/etc/docker/daemon.json文件：

```
vim /etc/docker/daemon.json
```

```
{
  "exec-opts": ["native.cgroupdriver=systemd"]
}
```

```
systemctl daemon-reload && systemctl restart docker

#查看当前管理器模式
docker info
```


在安装k8s集群时，需要把docker 的 cgroup 改成 systemd，修改完后报错。启动失败原因：直接启动docker会报错，因为docker.service里有一条配置，和刚才添加的"exec-opts"冲突了：


**解决报错:**

vi /lib/systemd/system/docker.service

找到并删除下面这句话，保存退出，即可解决

```
--exec-opt native.cgroupdriver=cgroupfs
```

```
systemctl daemon-reload
systemctl restart docker
```


