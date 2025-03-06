
## 下载 FTP


```
opkg update
opkg install vsftpd openssh-sftp-server
/etc/init.d/vsftpd enable
/etc/init.d/vsftpd start
```



## 修改配置

/etc/vsftpd.conf
```
secure_chroot_dir=/root
ftp_username=root
```

/etc/init.d/vsftpd restart
