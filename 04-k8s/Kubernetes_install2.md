# 云端部署

kubernetes 的 [安装过程](https://blog.csdn.net/weixin_39177986/article/details/124807924) 就不在这里细说了，想要了解的可以查看连接中的博客。


## 查看 kuberetes 工具是否安装

```
yum list installed | grep kubelet
yum list installed | grep kubeadm
yum list installed | grep kubectl
```

## 检查 MAC 与 product_uuid

使用命令 ip link 或 ifconfig -a 来获取网络接口的 MAC 地址
使用 sudo cat /sys/class/dmi/id/product_uuid 命令对 product_uuid 校验
一般来讲，硬件设备会拥有唯一的地址，但是有些虚拟机的地址可能会重复。 Kubernetes 使用这些值来唯一确定集群中的节点。 如果这些值在每个节点上不唯一，可能会导致安装失败

## 容器运行时

容器不光是 Docker，还有其他容器，比如 CoreOS 的 rkt。为了保证容器生态的健康发展，保证不同容器之间能够兼容，包含 Docker、CoreOS、Google在内的若干公司共同成立了一个叫 Open Container Initiative（OCI） 的组织，其目是制定开放的容器规范。

runtime 是容器真正运行的地方，runtime 需要跟操作系统 kernel 紧密协作，为容器提供运行环境。

lxc、runc 和 rkt 是目前主流的三种容器 runtime。

- lxc 是 Linux 上老牌的容器 runtime，Docker 最初也是用 lxc 作为 runtime。
- runc 是 Docker 自己开发的容器 runtime，符合 oci 规范，也是现在 Docker 的默认 runtime。
- rkt 是 CoreOS 开发的容器 runtime，符合 oci 规范，因而能够运行 Docker 的容器。


# 镜像拉取

## 查看需拉取镜像列表

kubeadm config images list

```
I0607 14:51:32.871878  275642 version.go:255] remote version is much newer: v1.24.1; falling back to: stable-1.22
k8s.gcr.io/kube-apiserver:v1.22.10
k8s.gcr.io/kube-controller-manager:v1.22.10
k8s.gcr.io/kube-scheduler:v1.22.10
k8s.gcr.io/kube-proxy:v1.22.10
k8s.gcr.io/pause:3.5
k8s.gcr.io/etcd:3.5.0-0
k8s.gcr.io/coredns/coredns:v1.8.4
```

## 默认镜像拉取方式

默认情况下, kubeadm 会从 k8s.gcr.io 仓库拉取镜像。如果请求的 Kubernetes 版本是 CI 标签 （例如 ci/latest），则使用 gcr.io/k8s-staging-ci-images。k8s.gcr.io 仓库需要使用外网，建议使用内网支持的镜像库。使用 dockerhub 下的 k8simage，这个域名下同步了不少谷歌镜像：

```
docker pull docker.io/k8simage/kube-apiserver:v1.22.10
docker pull docker.io/k8simage/kube-controller-manager:v1.22.10
docker pull docker.io/k8simage/kube-scheduler:v1.22.10
docker pull docker.io/k8simage/kube-proxy:v1.22.10
docker pull docker.io/k8simage/pause:3.5
docker pull docker.io/k8simage/etcd:3.5.0-0
docker pull docker.io/k8simage/coredns/coredns:v1.8.4
```

下载之后对镜像从新打标签:

```
docker tag docker.io/k8simage/kube-proxy-amd64:v1.11.3 k8s.gcr.io/kube-proxy-amd64:v1.11.3
```

当然，在配置文件中也可以通过修改 kubeadm 初始化配置进行指定下载镜像库，这是最简便的方式。

## 修改配置文件

获取默认配置文件并保存至本地

```
kubeadm config print init-defaults > kubeadm-defaults.yaml
```

可以通过修改配置文件 (kubeadm-defaults.yaml) 指定镜像下载链接，以下是本设备使用的一个配置文件内容，可用作与参考：

```
apiVersion: kubeadm.k8s.io/v1beta3
kind: InitConfiguration
bootstrapTokens:
  - groups:
      - system:bootstrappers:kubeadm:default-node-token
    token: abcdef.0123456789abcdef
    ttl: 24h0m0s
    usages:
      - signing
      - authentication
nodeRegistration:
  criSocket: /var/run/dockershim.sock
  imagePullPolicy: IfNotPresent
  taints: null
localAPIEndpoint:
  advertiseAddress: 172.16.16.6
  bindPort: 6443
---
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
etcd:
  local:
    dataDir: /var/lib/etcd
networking:
  dnsDomain: cluster.local
  podSubnet: "10.244.0.0/24"
  serviceSubnet: 10.96.0.0/12
apiServer:
  timeoutForControlPlane: 4m0s

certificatesDir: /etc/kubernetes/pki
clusterName: kubernetes
controllerManager: {}
dns: {}

imageRepository: docker.io/k8simage

#修改为当前使用版本号
kubernetesVersion: 1.22.9

scheduler: {}
---
#在 v1.22 中，如果用户没有设置cgroupDriver下的字段KubeletConfiguration， kubeadm将默认为systemd
kind: KubeletConfiguration
#apiVersion: kubelet.config.k8s.io/v1beta3
apiVersion: kubelet.config.k8s.io/v1beta1
#修改kubelet驱动
cgroupDriver: systemd
```
可以引用：
[kubeadm-defaults.yaml 配置文件样例](https://github.com/laneston/blog/blob/main/k8s/kubeadm-defaults.yaml)
并执行:

```
kubeadm init --config ./kubeadm-defaults.yaml
```


```
kubeadm init 
```

```
kubeadm init --kubernetes-version=1.22.9 \
--apiserver-advertise-address=172.16.16.6 \
--image-repository=docker.io/k8simage \
--pod-network-cidr=10.244.0.0/16
```



# 错误处理

## 问题01 ip_forward不为1

```
[ERROR FileContent--proc-sys-net-ipv4-ip_forward]: /proc/sys/net/ipv4/ip_forward contents are not set to 1
```
查看ip_forward：
```
cat /proc/sys/net/ipv4/ip_forward
```
确实不为1

设置：
```
echo 1 > /proc/sys/net/ipv4/ip_forward
```


## 问题02 初始化提示The kubelet is not running

```
   Unfortunately, an error has occurred:
                timed out waiting for the condition

        This error is likely caused by:
                - The kubelet is not running
                - The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)

        If you are on a systemd-powered system, you can try to troubleshoot the error with the following commands:
                - 'systemctl status kubelet'
                - 'journalctl -xeu kubelet'

        Additionally, a control plane component may have crashed or exited when started by the container runtime.
        To troubleshoot, list all containers using your preferred container runtimes CLI.

        Here is one example how you may list all Kubernetes containers running in docker:
                - 'docker ps -a | grep kube | grep -v pause'
                Once you have found the failing container, you can inspect its logs with:
                - 'docker logs CONTAINERID'

error execution phase wait-control-plane: couldn't initialize a Kubernetes cluster
To see the stack trace of this error execute with --v=5 or higher
```

配置文件中带有字段 name ，且与 node 不符合：
```
nodeRegistration:
  criSocket: /var/run/dockershim.sock
  imagePullPolicy: IfNotPresent
  name: node
  taints: null
```

## 问题03 Unable to update cni config" err="no networks found in /etc/cni/net.d

```
Jul 02 10:48:45 VM-0-4-centos kubelet[178540]: I0702 10:48:45.239162  178540 cni.go:239] "Unable to update cni config" err="no networks found in /etc/cni/net.d"
Jul 02 10:48:47 VM-0-4-centos kubelet[178540]: E0702 10:48:47.270449  178540 kubelet.go:2376] "Container runtime network not ready" networkReady="NetworkReady=false reason:NetworkPluginNotReady message:docker: network plugin is not ready: cni config uninitialized"
```

解决方案是 [安装 flannel 组件](#flannel)

## 问题03 networkPlugin cni failed to set up pod

```
"RunPodSandbox from runtime service failed" err="rpc error: code = Unknown desc = failed to set up sandbox container \"a735b231c34fed86068e61a4873bd8f4034fbd8e158765d62e86ca81646e6a51\" network for pod \"coredns-57bb6f6c5-79qjc\": networkPlugin cni failed to set up pod \"coredns-57bb6f6c5-z7zcc_kube-system\" network: open /run/flannel/subnet.env: no such file or directory"
```

原因是缺少：

```
networking:
  dnsDomain: cluster.local
  podSubnet: "10.244.0.0/16"
  serviceSubnet: 10.96.0.0/12
```


## 重置 kubeadm 配置

如果需要将 kubeadm 配置恢复到初始状态，可执行以下命令：

```
kubeadm reset

systemctl daemon-reload
systemctl restart kubelet.service
```

如果配置文件符合部署环境，则会看到如下打印，则说明初始化成功：


# 安装 flannel 组件
<a id='flannel'></a>

部署文件路径如下：https://github.com/flannel-io/flannel/blob/master/Documentation/kube-flannel.yml

将文件下载至本地后可以通过以下命令进行安装：
```
kubectl apply -f kube-flannel.yml
```

此时可能会弹出以下错误警告信息：

```
The connection to the server localhost:8080 was refused - did you specify the right host or port?
```

原因：kubernetes master没有与本机绑定，集群初始化的时候没有绑定，此时设置在本机的环境变量即可解决问题。

解决办法：
```
echo "export KUBECONFIG=/etc/kubernetes/admin.conf" >> /etc/profile
source /etc/profile
```

&&

```
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config && sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

再执行：
```
kubectl apply -f kube-flannel.yml
```

# 成功执行

```
[root@VM-16-6-centos home]# kubectl get pod -o wide -A              
NAMESPACE     NAME                                     READY   STATUS    RESTARTS   AGE
kube-system   coredns-57bb6f6c5-7jrt5                  1/1     Running   0          9m26s
kube-system   coredns-57bb6f6c5-tdnxx                  1/1     Running   0          9m26s
kube-system   etcd-vm-16-6-centos                      1/1     Running   1          9m40s
kube-system   kube-apiserver-vm-16-6-centos            1/1     Running   1          9m40s
kube-system   kube-controller-manager-vm-16-6-centos   1/1     Running   0          9m40s
kube-system   kube-flannel-ds-9r2g9                    1/1     Running   0          2m22s
kube-system   kube-proxy-qpqbm                         1/1     Running   0          9m26s
kube-system   kube-scheduler-vm-16-6-centos            1/1     Running   1          9m40s
```

kubeadm certs check-expiration

##  检查集群状态提示 10251 端口 connection refused

查看集群状态

```
kubectl get cs
```

```
Warning: v1 ComponentStatus is deprecated in v1.19+
NAME                 STATUS      MESSAGE                                                                                       ERROR
scheduler            Unhealthy   Get "http://127.0.0.1:10251/healthz": dial tcp 127.0.0.1:10251: connect: connection refused   
controller-manager   Healthy     ok                                                                                            
etcd-0               Healthy     {"health":"true","reason":""}      
```

这种情况是因为kube-controller-manager.yaml和kube-scheduler.yaml 里面配置了默认端口0，需要将两个文件中 --port=0 参数注释。


```
systemctl restart kubelet.service
```



kubectl delete po -A --all



```
.:53
[INFO] plugin/reload: Running configuration MD5 = db32ca3650231d74073ff4cf814959a7
CoreDNS-1.8.4
linux/amd64, go1.16.4, 053c4d5
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:57540->183.60.82.98:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:52866->183.60.83.19:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:41262->183.60.83.19:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:35873->183.60.82.98:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:40820->183.60.82.98:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:51626->183.60.83.19:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:34766->183.60.82.98:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:60418->183.60.82.98:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:51914->183.60.82.98:53: i/o timeout
[ERROR] plugin/errors: 2 1869938021561953328.7458345514577076202. HINFO: read udp 10.244.0.2:51287->183.60.83.19:53: i/o timeout
```
修改以下文件
vim /etc/resolv.conf

不能超出3个，多了需要删除
nameserver 223.6.6.6
nameserver 8.8.8.8

