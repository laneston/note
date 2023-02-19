# 什么是 flannel

Flannel是CoreOS团队针对Kubernetes设计的一个网络规划服务，运行在一个网上的网（应用层网络），并不依靠ip地址来传递消息，而是采用一种映射机制，把ip地址和identifiers做映射来资源定位。 简单来说，它的功能是让集群中的不同节点主机创建的Docker容器都具有全集群唯一的虚拟IP地址。

Flannel github：https://github.com/coreos/flannel

## 为什么使用 flannel

在默认的Docker配置中，每个节点上的Docker服务会分别负责所在节点容器的IP分配。这样导致的一个问题是，不同节点上容器可能获得相同的内外IP地址。

Flannel的设计目的就是为集群中的所有节点重新规划IP地址的使用规则，从而使得不同节点上的容器能够获得“同属一个内网”且”不重复的”IP地址，并让属于不同节点上的容器能够直接通过内网IP通信。

## 如何实现 flannel

Flannel实质上是一种“覆盖网络(overlay network)”，也就是将TCP数据包装在另一种网络包里面进行路由转发和通信，目前已经支持UDP、VxLAN、AWS VPC和GCE路由等数据转发方式，默认的节点间数据通信方式是UDP转发。

# 安装 flannel

-------------------

## 安装 Etcd

因为 flannel 依赖 etcd，所以需要安装 etcd。本次使用的安装包为： etcd-v3.5.4-linux-amd64.tar.gz



```
# 解压安装包
tar -zxvf etcd-v3.5.4-linux-amd64.tar.gz
cd etcd-v3.5.4-linux-amd64
cp etcd* /usr/bin/
```

```
mkdir /etc/etcd
vim /etc/etcd/etcd.conf
```

```
ETCD_NAME=ETCD_1
ETCD_DATA_DIR="/var/lib/etcd/"
ETCD_LISTEN_CLIENT_URLS="http://106.52.179.147:2379"
ETCD_ADVERTISE_CLIENT_URLS="http://106.52.179.147:2379"
```

/usr/lib/systemd/system/etcd.service
```
[Unit]
Description=Etcd Server
After=network.target

[Service]
Type=simple
TimeoutStartSec=1
WorkingDirectory=/var/lib/etcd/
EnvironmentFile=-/etc/etcd/etcd.conf
ExecStart=/usr/bin/etcd

[Install]
WantedBy=multi-user.target
```

重新加载某个服务的配置文件，如果新安装了一个服务，归属于 systemctl 管理，要是新服务的服务程序配置文件生效，需重新加载:


```
systemctl daemon-reload
systemctl enable etcd.service
systemctl start etcd.service
```

到此 Etcd 安装结束。


-----------------

## 在线安装

```
yum install -y flannel
```

## 离线安装

下载安装包 flannel-v0.17.0-linux-amd64.tar.gz

```
cp cp flanneld /usr/bin/
```

vim /usr/lib/systemd/system/flanneld.service

```
[Unit]
Description=Flanneld overlay address etcd agent
After=network.target
After=network-online.target
Wants=network-online.target
After=etcd.service
Before=docker.service

[Service]
Type=notify
EnvironmentFile=/etc/sysconfig/flanneld
EnvironmentFile=-/etc/sysconfig/docker-network
ExecStart=/usr/bin/flanneld-start \
  -etcd-endpoints=${FLANNEL_ETCD_ENDPOINTS} \
  -etcd-prefix=${FLANNEL_ETCD_PREFIX} \
  $FLANNEL_OPTIONS
ExecStartPost=/usr/libexec/flannel/mk-docker-opts.sh -k DOCKER_NETWORK_OPTIONS -d /run/flannel/docker
Restart=on-failure

[Install]
WantedBy=multi-user.target
RequiredBy=docker.service
```

```
cd /usr/libexec & mkdir flannel
cd flannel
cp /home/flannel_0.17.0/mk-docker-opts.sh ./
```


vim /etc/sysconfig/flanneld

```
# Flanneld configuration options  

# etcd url location.  Point this to the server where etcd runs
FLANNEL_ETCD_ENDPOINTS="http://106.52.179.147:2379"

# etcd config key.  This is the configuration key that flannel queries
# For address range assignment
FLANNEL_ETCD_PREFIX="/kube-centos/network"

# Any additional options that you want to pass
FLANNEL_OPTIONS="-etcd-cafile=/etc/kubernetes/ssl/ca.pem -etcd-certfile=/etc/kubernetes/ssl/kubernetes.pem -etcd-keyfile=/etc/kubernetes/ssl/kubernetes-key.pem"
```




CNI 全称是 Container Network Interface，即容器网络的 API 接口。实现了这个接口的就是 CNI 插件，它实现了一系列的 CNI API 接口。常见的 CNI 插件包括 Calico、flannel、Terway、Weave Net 以及 Contiv。


K8s 通过 CNI 配置文件来决定使用什么 CNI。


## 使用 kubectl 在线安装

```
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml

```
## 使用 kubectl 离线安装

```
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```




