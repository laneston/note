# 用户管理

## 创建用户


```
# 创建用户
useradd cuinew
# 创建密码
passwd cuinew
```

## 赋予普通启用户root权限

### 更改配置文件 sudoers 的权限

```
[root@VM-0-4-centos home]# ll /etc/sudoers
-r--r-----. 1 root root 4328 Oct 25  2021 /etc/sudoers
```

原本 sudoers 是只读权限


```
chmod 777 /etc/sudoers
```

将文件修改为可读写文件即可实现赋权，但这种操作具有安全隐患，此时所有用户都可进行 root 操作。

- r:4 
- w:2 
- x:1

### 修改配置文件赋予 root 权限

赋权
```
[root@VM-0-4-centos home]# chmod 777 /etc/sudoers
```


编辑 sudoers 文件:
```
vim /etc/sudoers
```

增加一行 lanceli 权限开启选项
```
## Next comes the main part: which users can run what software on 
## which machines (the sudoers file can be shared between multiple
## systems).
## Syntax:
##
##      user    MACHINE=COMMANDS
##
## The COMMANDS section may have other options added to it.
##
## Allow root to run any commands anywhere 
root    ALL=(ALL)       ALL
lanceli ALL=(ALL)       ALL
```

重新赋权
```
[root@VM-0-4-centos home]# chmod 440 /etc/sudoers
[root@VM-0-4-centos home]# 
[root@VM-0-4-centos home]# 
[root@VM-0-4-centos home]# ll /etc/sudoers       
-r--r-----. 1 root root 4355 Aug  2 15:35 /etc/sudoers
```

到此完成。



# 文件类型

- b  代表块设备文件（经常使用的U盘）
- d  代表目录文件
- I  代表连接文件
- s  代表字连接（主要用于网络通信）
- p  代表管道文件（主要用于进程通信相关）
