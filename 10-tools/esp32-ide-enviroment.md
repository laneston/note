# ESP32 在 WIN10 系统上的开发环境搭建 

**IDE：** vscode

**依赖工具：** Python3.8

**开发工具：** ESP-IDF

由于本人习惯的编码 IDE 工具是 vscode，且 vscode 也有相关的插件支持，所以直接在 vscode 上加载ESP32的开发工具。搭建WIN10上的vscode开发ESP32的环境需要分三步走。

## 第一步：安装Python3.8

因为 vscode 的 esp 插件依赖 python，且版本最好为 3.8 的版本。

**安装链接：** <a href = "https://www.python.org/downloads/release/python-384/">Python 3.8.4</a>

pip 是 Python 包管理工具，该工具提供了对Python 包的查找、下载、安装、卸载的功能，更新pip:

```
python -m pip install --upgrade pip
```

### 安装错误

如果没有对默认的 PIP 源进行更改，在进行 vscode 安装 ESP-IDF 插件时会出现如下警告：

```
Command failed: "C:\Users\Administrator\.espressif\python_env\idf4.3_py3.8_env\Scripts\python.exe" -m pip install --upgrade --no-warn-script-location  -r "c:\Users\Administrator\.vscode\extensions\espressif.esp-idf-extension-1.1.1\requirements.txt"
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1125)'))': /simple/gcovr/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1125)'))': /simple/gcovr/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1125)'))': /simple/gcovr/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1125)'))': /simple/gcovr/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1125)'))': /simple/gcovr/
ERROR: Could not find a version that satisfies the requirement gcovr
ERROR: No matching distribution found for gcovr
WARNING: You are using pip version 20.3.3; however, version 21.2.4 is available.
You should consider upgrading via the 'C:\Users\Administrator\.espressif\python_env\idf4.3_py3.8_env\Scripts\python.exe -m pip install --upgrade pip' command.
```

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-ESP_Enviroment/ESP-IDF_warning.jpg"></div>

原因是默认的 pip 源加载不畅，解决办法是加载国内的 pip 源:

```
pip install keras -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

其余的源链接如下所示：

**阿里云** http://mirrors.aliyun.com/pypi/simple/

**中国科技大学** https://pypi.mirrors.ustc.edu.cn/simple/

**豆瓣(douban)** http://pypi.douban.com/simple/

**清华大学** https://pypi.tuna.tsinghua.edu.cn/simple/

**中国科学技术大学** http://pypi.mirrors.ustc.edu.cn/simple/

## 第二步：安装ESP-IDF

**安装链接：** <a href = "https://dl.espressif.com/dl/esp-idf/">ESP-IDF</a>

第一个是在线安装，如果网速不快，亦或是下载失败，可以选择第二个离线安装。

## 第三步：安装vscode插件

这一步比较简单，就是在vscode插件搜索栏中输入 ESP 然后选择对应图标的那个。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-ESP_Enviroment/ESP-VScode-plug-in.jpg"></div>

插件安装完毕后会进入 ESP-IDF Setup 界面，界面中有3个下拉选择栏，Select ESP-IDF version一栏中选择本地的 ESP-IDF(Find ESP-IDF in your system)；Enter ESP-IDF directory 一栏中，选择第二步中ESP-IDF安装的路径，文件夹的名字一般为 esp-idf，这个文件夹里面有相关例程可以用作测试。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-ESP_Enviroment/esp-installed.jpg"></div>

安装成功如上图显示。

打开路径 \esp-idf\examples\get-started\hello_world ，点击下图中的编译按键，即可进行编译。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-ESP_Enviroment/ESP-compile.jpg"></div>

编译完毕后，可以在 build 文件夹下看到相应的 bin 文件。

<div align="center"><img src="https://github.com/laneston/Pictures/blob/master/Post-ESP_Enviroment/ESP-Bin.jpg"></div>

将开发板与电脑相连接，点击闪电形状的下载按键即可进行烧录操作。
