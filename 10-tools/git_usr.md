# git账号设置

```
git config --global user.name "laneston"
git config --global user.email "lijiangliang2016@163.com"
```

# 配置远程仓库免登陆

```
ssh-keygen -t rsa -C "lijiangliang2016@163.com"
```

密钥默认存放在用户目录下。

将公钥添加到 gitHub 的 SSH key当中。 

```
cat id_rsa.pub
```

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-git_usr/ssh_key.png"></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-git_usr/key_item.png"></div>

测试是否能够连接到 github 网站：

```
ssh -T git@github.com
```
