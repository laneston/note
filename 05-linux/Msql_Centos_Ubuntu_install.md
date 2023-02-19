# Mysql 在 ubuntu20 和 Centos7 上的编译安装


## 编译

下载 Mysql 安装包，下载连接：https://downloads.mysql.com/archives/community/

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Msql_Centos_Ubuntu_install/mysql_download1.jpg"></div>


目前使用的版本是： 5.7.32 下载时有带 boost 库的，也有不带 boost 库的，对本次编译的差别不大，本次编译选用的是不带 boost 的，会在编译的过程中自己下载相应的文件。

执行 tar zxvf mysql-5.7.32.tar.gz 命令进行解压，权限不够就在前面加个 sudo

解压之后进入编译文件的 1 级目录，执行以下 cmake 命令：

```
cmake . -DDOWNLOAD_BOOST=1 -DWITH_BOOST=/home/mysql-5.7.32/boost -DCMAKE_INSTALL_PREFIX=/home/mysql-5.7.32/__install
```

1. -DDOWNLOAD_BOOST=1 表示使能 BOOST 文件的下载；
2. -DWITH_BOOST=/home/mysql-5.7.32/boost 表示BOOST 文件的下载路径；
3. -DCMAKE_INSTALL_PREFIX=/home/mysql-5.7.32/__install 表示编译完毕后的 mysql 文件安装的路径。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Msql_Centos_Ubuntu_install/mysql_download2.jpg"></div>


如果出现警告，则按相应提示进行软件的安装。

准备完毕后，输入 make 进行编译，编译完毕后，输入 make install进行安装，那么相关文件都会在刚才指定的 __install 目录下。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Msql_Centos_Ubuntu_install/mysql_download3.jpg"></div>

## 安装

将文件复制到要安装的目录下，本人安装的路径为 /usr/local/mysql

复制完毕后需要进行一下几步操作对 mysql 进行设置，使之能正常工作。


### 设置环境变量

在环境变量文件 /etc/profile 最后添加以下语句

```
export PATH=/usr/local/mysql/bin:$PATH
```

执行 source /etc/profile 同步

在动态库链接配置文件 /etc/ld.so.conf 中添加 lib 文件夹的路径，然后执行 /sbin/ldconfig 同步。

### 创建用户

```
root@Lance-PC:/usr/local# useradd mysql
root@Lance-PC:/usr/local# passwd mysql
New password:
Retype new password:
passwd: password updated successfully
```

使用以上命令创建 mysql 用户。

### 目录操作

生成以下路径下的目录： /usr/local/mysql_database/data

分别对安装目录与 mysql_database 目录设置权限：

```
root@Lance-PC:/usr/local# chown -R mysql:mysql mysql_database/
root@Lance-PC:/usr/local# chown -R mysql:mysql mysql

```

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Msql_Centos_Ubuntu_install/mysql_download4.jpg"></div>


### 添加配置文件

在 /etc 目录下添加 my.cnf 文件，文件内容如下，参数值需要根据自己实际路径进行配置，如果安装过程中有错误，请检查内容是否与配置时的命令相符。

```
[client]
port        = 3306
socket      = /tmp/mysql.sock

[mysqld]
port        = 3306
socket      = /tmp/mysql.sock
user = mysql

basedir = /usr/local/mysql
datadir = /usr/local/mysql_database/data
pid-file = /usr/local/mysql_database/mysql.pid

log_error = /usr/local/mysql_database/mysql-error.log
slow_query_log = 1
long_query_time = 1
slow_query_log_file = /usr/local/mysql_database/mysql-slow.log

#skip-grant-tables

skip-external-locking
key_buffer_size = 32M
max_allowed_packet = 1024M
table_open_cache = 128
sort_buffer_size = 768K
net_buffer_length = 8K
read_buffer_size = 768K
read_rnd_buffer_size = 512K
myisam_sort_buffer_size = 8M
thread_cache_size = 16
query_cache_size = 16M
tmp_table_size = 32M
performance_schema_max_table_instances = 1000

explicit_defaults_for_timestamp = true

#skip-networking
max_connections = 500
max_connect_errors = 100
open_files_limit = 65535

log_bin=mysql-bin
binlog_format=mixed
server_id   = 232
expire_logs_days = 10
early-plugin-load = ""

default_storage_engine = InnoDB
innodb_file_per_table = 1
innodb_buffer_pool_size = 128M
innodb_log_file_size = 32M
innodb_log_buffer_size = 8M
innodb_flush_log_at_trx_commit = 1
innodb_lock_wait_timeout = 50

[mysqldump]
quick
max_allowed_packet = 16M

[mysql]
no-auto-rehash

[myisamchk]
key_buffer_size = 32M
sort_buffer_size = 768K
read_buffer = 2M
write_buffer = 2M
```

### 初始化mysql

```
mysqld --initialize-insecure --user=mysql --basedir=/usr/local/mysql --datadir=/usr/local/mysql_database/data
```

使用以上命令对 mysql 进行初始化，注意路径是否与 my.cnf 文件中的相同。

### 拷贝可执行配置文件

```
root@Lance-PC:/usr/local# cp /usr/local/mysql/support-files/mysql.server /etc/init.d/mysqld
```

### 启动 mysql

```
service mysqld start
```

### 连接 mysql

```
mysql -u root -p
```

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Msql_Centos_Ubuntu_install/mysql_download5.jpg"></div>

连接过程中可能会出现以上情况。

解决方案如下：

将 my.cnf 中 skip-grant-tables 语句的注释取消，执行 /etc/init.d/mysqld restart 重启 mysql。此时可以不输密码进入数据库，输入 mysql -u root -p 后按回车即可。


<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Msql_Centos_Ubuntu_install/mysql_download6.jpg"></div>

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-Msql_Centos_Ubuntu_install/mysql_download7.jpg"></div>


可以看到 root 用户没有设置密码，这个原因导致了这个错误。


```
mysql > update user set authentication_string=password('xxxxxxx') where user='root';
```

可以使用以上命令来对 root 用户设置密码，设置完毕后可以输入 qiut; 或 \q 退出。


将 my.cnf 中 skip-grant-tables 语句的注释加上，重启 mysql 即可。


## 设置开启自启动

```
systemctl enable mysqld
```

这个选项看需求设置。



通过以上操作，mysql 就已经安装完毕可以使用了。

