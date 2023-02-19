# 概述

## kubernetes 与 kubeEdge 对应关系

|                        | Kubernetes 1.16 | Kubernetes 1.17 | Kubernetes 1.18 | Kubernetes 1.19 | Kubernetes 1.20 | Kubernetes 1.21 | Kubernetes 1.22 |
| ---------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| KubeEdge 1.8           | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               |
| KubeEdge 1.9           | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               |
| KubeEdge 1.10          | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               |
| KubeEdge HEAD (master) | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               | ✓               |

## 为什么要部署kubeEdge

1. 边缘侧设备没有足够的资源运行一个完整的 Kubelet，Kubelet主要作用是获取最新的规范，确保各节点的 Pod 和容器在规范下运行；
2. 某些边缘侧设备是ARM架构，然而kubernetes不支持ARM架构。

1. KubeEdge 保留了 Kubernetes 的管理面，重新开发了节点 agent，大幅度优化让边缘组件资源占用更低很多；
2. KubeEdge 可以完美支持 ARM 架构和 x86 架构；
3. KubeEdge 有离线自治功能；
4. KubeEdge 丰富了应用和协议支持，目前已经支持和计划支持的有：MQTT、BlueTooth、OPC UA、Modbus等；
5. KubeEdge 通过底层优化的多路复用消息通道优化了云边的通信的性能。


## 污点容忍设置


如若以下错误，则需要清除污点操作。
```
Kubernetes version verification passed, KubeEdge installation will start...
Error: timed out waiting for the condition
```





查看污点

```
kubectl describe nodes vm-16-6-centos | grep Taints
```

清除污点

```
kubectl taint node [nodeName] node-role.kubernetes.io/master-
```
or
```
kubectl taint nodes --all node-role.kubernetes.io/master-
```





# 部署步骤

## cloudcore 部署

```
mkdir /etc/kubeedge/ && cd /etc/kubeedge/
```


将文件放入文件夹内

```
./keadm init --advertise-address=106.52.34.106 --kubeedge-version=1.10.3  --kube-config=/root/.kube/config
```

```
./keadm deprecated init --advertise-address=106.52.34.106 --kubeedge-version=1.12.1  --kube-config=/root/.kube/config
```


```
sudo cp /etc/kubeedge/cloudcore.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cloudcore.service
sudo systemctl start cloudcore.service
```




journalctl -u cloudcore.service -f





listen tcp 0.0.0.0:10002: bind: address already in use
端口号被占用
重启服务器


export CLOUDCOREIPS="106.52.34.106"
cd /etc/kubeedge/ && chmod +x certgen.sh
./certgen.sh stream

[root@VM-16-6-centos kubeedge]# kubectl get cm tunnelport -nkubeedge -oyaml
apiVersion: v1
kind: ConfigMap
metadata:
  annotations:
    tunnelportrecord.kubeedge.io: '{"ipTunnelPort":{"172.16.16.6":10351},"port":{"10351":true}}'
  creationTimestamp: "2022-12-07T09:34:08Z"
  name: tunnelport
  namespace: kubeedge
  resourceVersion: "3326"
  uid: d0cf19f8-2d1f-4485-adcb-982cd4111dfb


iptables -t nat -A OUTPUT -p tcp --dport 10350 -j DNAT --to 172.16.16.6:10003
iptables -t nat -A OUTPUT -p tcp --dport 10351 -j DNAT --to 172.16.16.6:10003

如果设置错误，可使用以下命令进行删除
iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X

修改 cloudcore.yaml文件
sudo vim /etc/kubeedge/config/cloudcore.yaml

```
edgeStream:
  enable: true
  handshakeTimeout: 30
  readDeadline: 15
  server: 192.168.0.139:10004
  tlsTunnelCAFile: /etc/kubeedge/ca/rootCA.crt
  tlsTunnelCertFile: /etc/kubeedge/certs/server.crt
  tlsTunnelPrivateKeyFile: /etc/kubeedge/certs/server.key
  writeDeadline: 15
  ```

  设置 enable 为 true

systemctl restart cloudcore.service

ps -ef|grep cloudcore

journalctl -u cloudcore.service -xe
journalctl -u cloudcore.service -f



kubectl edit daemonsets.apps -n kube-system kube-proxy



```
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
          - key: node-role.kubernetes.io/edge
            operator: DoesNotExist
```

```
./keadm gettoken --kube-config=$HOME/.kube/config
```

## edgecore 部署

```
mkdir /etc/kubeedge/ && cd /etc/kubeedge/
```

```
keadm join --cloudcore-ipport=106.52.34.106:10000 --edgenode-name=ins-q8we81tu --kubeedge-version=1.10.3 --with-mqtt=false --token=24eb6189eba66f2854a7bc299d864ff4123946035d38b6204a58b09256ff9191.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2NzM5MTUyNjB9.8JFf0SPx0s6d8iprZ1EZdNNTdnGkQqWWvXVLrEYMHlA
```



keadm join --cloudcore-ipport=106.52.34.106:10000 --edgenode-name=mtk100 --kubeedge-version=1.10.3 --with-mqtt=false --token=24eb6189eba66f2854a7bc299d864ff4123946035d38b6204a58b09256ff9191.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2NzM5MTUyNjB9.8JFf0SPx0s6d8iprZ1EZdNNTdnGkQqWWvXVLrEYMHlA





sudo cp /etc/kubeedge/edgecore.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edgecore.service
sudo systemctl start edgecore.service


sudo vim /etc/kubeedge/config/edgecore.yaml

```
edgeStream:
  enable: true
  handshakeTimeout: 30
  readDeadline: 15
  server: 192.168.0.139:10004
  tlsTunnelCAFile: /etc/kubeedge/ca/rootCA.crt
  tlsTunnelCertFile: /etc/kubeedge/certs/server.crt
  tlsTunnelPrivateKeyFile: /etc/kubeedge/certs/server.key
  writeDeadline: 15
```


ps -ef|grep edgecore

systemctl restart edgecore.service
journalctl -u edgecore.service -f

[Deploy demo on edge nodes](https://kubeedge.io/en/docs/setup/keadm/)







# CA 证书制作

```
cat > ca-csr.json <<EOF
{
	"CN": "Kubernetes",
	"key": {
		"algo": "rsa",
		"size": 2048
	},
	"names": [
		{
			"C": "CN",
			"ST": "GD",
			"L": "ShenZhen",
			"O": "Kubernetes",
			"OU": "CA"
		}
	]
}
EOF
```


```
cfssl gencert -initca ca-csr.json | cfssljson -bare ca
```


```
cat > ca-config.json <<EOF
{
	"signing": {
		"default": {
			"expiry": "876000h"
		},
		"profiles": {
			"kubernetes": {
				"usages": [
					"signing",
					"key encipherment",
					"server auth",
					"client auth"
				],
				"expiry": "876000h"
			}
		}
	}
}
EOF
```


```
cat > kubeedge-csr.json <<EOF
{
	"CN": "kubeedge",
	"hosts": [		
		"*.com",
		"*.com.cn",
		"*.cn",
		"*.kubeedge.cn",
		"*.kubeedge.com",
		"10.244.0.1"
	],
	"key": {
		"algo": "rsa",
		"size": 2048
	},
	"names": [
		{
			"C": "CN",
			"ST": "GD",
			"L": "ShenZhen",
			"O": "kubeedge",
			"OU": "kubeedge"
		}
	]
}
EOF
```

```
cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -profile=kubernetes kubeedge-csr.json | cfssljson -bare kubeedge
```

```
cp ca.pem /etc/kubeedge/ca/rootCA.crt
cp ca-key.pem /etc/kubeedge/ca/rootCA.key
cp kubeedge.pem /etc/kubeedge/certs/server.crt
cp kubeedge-key.pem /etc/kubeedge/certs/server.key
```

# 删除节点

```
kubectl cordon <node name>
kubectl drain <node name> --ignore-daemonsets
kubectl delete node
```

kubectl cordon ins-q8we81tu
kubectl drain ins-q8we81tu --delete-local-data --ignore-daemonsets
kubectl delete node ins-q8we81tu




# 删除部署

```
kubectl delete deployments podtest-deployment
```

# 屏蔽kube-proxy

```
kubectl edit daemonsets.apps -n kube-system kube-proxy
kubectl edit daemonsets.apps -n kube-system kube-flannel
```


kubectl get secret tokensecret -n kubeedge -oyaml
kubectl delete secret casecret cloudcoresecret -n kubeedge
