# rabbitMQ 镜像部署

常用端口描述：

- 端口 15672 为网页管理端口；
- 端口 5672 为 AMQP 端口。
  

# 启动与使用




```
docker run -d --hostname my-rabbit --name rabbit -p 15672:15672 -p 5672:5672 rabbitmq
docker exec -it contain-id /bin/bssh
rabbitmq-plugins enable rabbitmq_management
```

默认账号密码：

```
admin: guest
password: guest
```




