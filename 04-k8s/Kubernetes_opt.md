
# 加入集群操作

## 查看 kubeadm join 参数


discovery-token-ca-cert-hash:
```
openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -hex | sed 's/^.* //'
```

token:
```
kubeadm token list
```

## 加入集群

```
kubeadm join 172.16.16.6:6443 --discovery-token abcdef.0123456789abcdef --discovery-token-ca-cert-hash sha256:48a556c897148d5a54c83fba948e551f009b1ce45f1fccfa74948c93285d137b
```

# 使用公网IP进行集群操作

回顾 kubernetes 服务端认证的流程: 任意一个客户端想访问 kube-apiserver 时，要拿服务器证书进行解析，去看 kube-apiserver 都有哪些别名，再把客户端访问 kube-apiserver 时所采用的 IP 或者域名和别名相比较，看是否已经涵盖在别名里面了。如果涵盖进去，就认为这个server是可认证的。由于现在已经部署出来的集群中，**Kubeadm 生成 APIServer 证书时没有把公网 IP 写到证书里**，所以导致用公网 IP访问不通过验证。默认情况下已经签发的证书表明，可访问的是 k8s 集群监听的内网 IP、kubernetes、kubernetes.default kubernetes.default.svc、kubernetes.default.svc.cluster.local，但不包含集群的外网 IP。

解决方案很简单，把需要用到的公网 IP 签到 API Server 证书里面就可以了，既可以通过命令行方式完成写入，参数名称是 --apiserver-cert-extra-sans，也可以通过 kubeadm 的 yaml 资源清单写入，资源清单中这个选项叫 apiServer -> CertSANs。只要把需要的公网 IP 签到 kube-apiserver 的证书中，再启动这个集群的时候，外部就可以通过集群的公网IP + 端口 与kube-apiserver进行认证了。


```
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
etcd:
  # one of local or external
  local:
    imageRepository: "docker.io/k8simage"
    dataDir: "/var/lib/etcd"
    extraArgs:
      listen-client-urls: "http://106.52.34.106:2379"
    serverCertSANs:
      - "106.52.34.106"
    peerCertSANs:
      - "106.52.34.106"
networking:
  dnsDomain: cluster.local
  podSubnet: "10.244.0.0/16"
  serviceSubnet: 10.96.0.0/12
apiServer:
  certSANs:
    - "106.52.34.106"
  timeoutForControlPlane: 4m0s

certificatesDir: /etc/kubernetes/pki
clusterName: kubernetes
controllerManager: {}
dns: {}

imageRepository: docker.io/k8simage

#修改为当前使用版本号
kubernetesVersion: 1.22.9

scheduler: {}
```


# 错误警告 CrashLoopBackOff

```
NAMESPACE     NAME                                     READY   STATUS             RESTARTS        AGE
kube-system   coredns-57bb6f6c5-b668n                  1/1     Running            1 (8m15s ago)   27m
kube-system   coredns-57bb6f6c5-nlr54                  1/1     Running            1 (8m15s ago)   27m
kube-system   etcd-vm-16-6-centos                      1/1     Running            1 (8m20s ago)   27m
kube-system   kube-apiserver-vm-16-6-centos            1/1     Running            8 (8m9s ago)    27m
kube-system   kube-controller-manager-vm-16-6-centos   1/1     Running            2 (8m20s ago)   27m
kube-system   kube-flannel-ds-csxb6                    0/1     CrashLoopBackOff   7 (16s ago)     12m
kube-system   kube-flannel-ds-pbgqc                    1/1     Running            1 (8m20s ago)   24m
kube-system   kube-proxy-g5g5g                         1/1     Running            1 (8m20s ago)   27m
kube-system   kube-proxy-twmjr                         1/1     Running            0               12m
kube-system   kube-scheduler-vm-16-6-centos            1/1     Running            2 (8m19s ago)   27m
```

查询原因：


```
[root@VM-16-6-centos ~]# kubectl -n kube-system logs kube-flannel-ds-csxb6
Error from server: Get "https://172.16.0.4:10250/containerLogs/kube-system/kube-flannel-ds-csxb6/kube-flannel": dial tcp 172.16.0.4:10250: connect: connection refused
```

kubeadm-defaults.yaml 文件中 
```
podSubnet: "10.244.0.0/24" 
```
与 kube-flannel.yml 文件中
```
  net-conf.json: |
    {
      "Network": "10.244.0.0/16",
      "Backend": {
        "Type": "vxlan"
      }
    }
```
不相同，将 podSubnet: "10.244.0.0/16" 修改为与 kube-flannel.yml 文件相同即可。


 