在执行 "make bananapi_r64_defconfig" 命令时，出现以下错误警告：

```
  YACC    scripts/kconfig/zconf.tab.c
/bin/sh: 1: bison: not found
make[1]: *** [scripts/Makefile.lib:222: scripts/kconfig/zconf.tab.c] Error 127
make: *** [Makefile:565: bananapi_r64_defconfig] Error 2
```

解决办法为安装以下两个工具即可：

```
apt-get install bison
apt-get install flex
```