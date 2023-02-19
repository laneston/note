10250 端口监听的是 kubelet 的 API 接口，是 kubelet 与 apiserver 通信的端口。kubelet 通过 10250 端口请求 apiserver 获取自己所应当处理的任务，并通过该端口访问及获取 node 资源以及状态。kubectl 查看 pod 的日志和 cmd 命令，都是通过 kubelet 端口 10250 访问。


