
本篇笔记记录的是参考 Tinyhttpd 来做的一个 HTTP server ，源工程 CGI 使用了 PERL 作为脚本语言，使得给嵌入式平台移植带来了额外的工作量，我写这笔记的目的是学习与记录如何编写一个基于 C 实现的 HTTP Server 和 CGI 程序，为本人的  [进程管理与进程间消息路由的轻量级项目框架](https://github.com/laneston/brick) 添砖加瓦，实现网关端 WEB 页面配置的功能。

# socket init

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-http_c/nng_http.png"></div>

```
static int startup(u_short *port)
{
    int httpd = 0;
    int on = 1;
    struct sockaddr_in name;

    assert(port != NULL);

    /// 1. socket init.
    httpd = socket(PF_INET, SOCK_STREAM, 0);
    if (httpd == -1)
        error_die("socket");

    memset(&name, 0, sizeof(name));
    name.sin_family = AF_INET;
    name.sin_port = htons(*port); // 将端口号由主机字节序转换为网络字节序的整数值
    name.sin_addr.s_addr = htonl(INADDR_ANY);
    if ((setsockopt(httpd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on))) < 0)
    {
        error_die("setsockopt failed");
    }

    /// 2. socket bind.
    if (bind(httpd, (struct sockaddr *)&name, sizeof(name)) < 0)
        error_die("bind");

    if (*port == 0) /* if dynamically allocating a port */
    {
        socklen_t namelen = sizeof(name);
        if (getsockname(httpd, (struct sockaddr *)&name, &namelen) == -1)
            error_die("getsockname");
        *port = ntohs(name.sin_port);
    }

    /// 3. socket listen.
    if (listen(httpd, 5) < 0)
        error_die("listen");

    return (httpd);
}

/**
 * httpd start function.
 */
int httpd_init(void)
{

    
    u_short port = 4000;
    int client_sock = -1;
    struct sockaddr_in client_name;
    socklen_t client_name_len = sizeof(client_name);
    pthread_t newthread;

    server_sock = startup(&port);
    log_printf("httpd_init >> httpd running on port %d\n", port);

    while (true)
    {
        /// 4. socket accept.
        client_sock = accept(server_sock, (struct sockaddr *)&client_name, &client_name_len);
        printf("accept >> client_sock: %d\n", client_sock);

        if (client_sock == -1)
            error_die("accept");

        /* accept_request(&client_sock); */
        if (pthread_create(&newthread, NULL, (void *)accept_request, (void *)(intptr_t)client_sock) != 0)
            perror("pthread_create");
    }

    return 0;
}
```


1. 服务器启动，在指定端口或随机选取端口绑定 httpd 服务。

2. 收到一个 HTTP 请求时（其实就是 listen 的端口 accpet 的时候），派生一个线程运行 accept_request 函数。

# accept request

```
void accept_request(void *arg)
{
    int client = (intptr_t)arg;
    char buf[1024];
    size_t numchars;
    char method[255];
    char url[255];
    char path[512];
    size_t i, j;
    struct stat st;
    int cgi = 0; /* becomes true if server decides this is a CGI
                  * program */
    char *query_string = NULL;

	/**取出 HTTP 请求中的 method (GET 或 POST) 和 url*/

    numchars = get_line(client, buf, sizeof(buf));

    i = 0;
    j = 0;

    while (!ISspace(buf[i]) && (i < sizeof(method) - 1))
    {
        method[i] = buf[i];
        i++;
    }

    j = i;

    method[i] = '\0';

    if (strcasecmp(method, "GET") && strcasecmp(method, "POST"))
    {
        unimplemented(client);
        return;
    }

    if (strcasecmp(method, "POST") == 0)
        cgi = 1;

    i = 0;

    while (ISspace(buf[j]) && (j < numchars))
        j++;

    while (!ISspace(buf[j]) && (i < sizeof(url) - 1) && (j < numchars))
    {
        url[i] = buf[j];
        i++;
        j++;
    }

    url[i] = '\0';


	/**对于 GET 方法，如果有携带参数，则 query_string 指针指向 url 中 ？ 后面的 GET 参数。*/

    if (strcasecmp(method, "GET") == 0)
    {
        query_string = url;

        while ((*query_string != '?') && (*query_string != '\0'))
            query_string++;

        if (*query_string == '?')
        {
            cgi = 1;
            *query_string = '\0';
            query_string++;
        }
    }

	/**格式化 url 到 path 数组，表示浏览器请求的服务器文件路径，在 tinyhttpd 中服务器文件是在 htdocs 文件夹下*/

    sprintf(path, "htdocs%s", url);

	/**当 url 以 / 结尾，或 url 是个目录，则默认在 path 中加上 index.html，表示访问主页*/

    if (path[strlen(path) - 1] == '/')
        strcat(path, "index.html");

    printf("accept_request >> %s\n", path);

	/**如果文件路径合法，对于无参数的 GET 请求，直接输出服务器文件到浏览器，即用 HTTP 格式写到套接字上，关闭浏览器连接*/
    if (stat(path, &st) == -1)
    {
        while ((numchars > 0) && strcmp("\n", buf)) /* read & discard headers */
            numchars = get_line(client, buf, sizeof(buf));

        not_found(client);
    }
    else
    {
		//如果文件是目录
        if ((st.st_mode & S_IFMT) == S_IFDIR)
        {
            strcat(path, "/index.html");
            printf("accept_request >>> %s\n", path);
        }

		//如果文件可执行
        if ((st.st_mode & S_IXUSR) || (st.st_mode & S_IXGRP) || (st.st_mode & S_IXOTH))
        {
            cgi = 1;
            printf("accept_request >> cgi = 1");
        }
		
		/**读取整个 HTTP 请求并丢弃，如果是 POST 则找出 Content-Length. 把 HTTP 200  状态码写到套接字*/
        if (!cgi)
        {
            serve_file(client, path);
        }
		/**其他情况（带参数 GET，POST 方式，url 为可执行文件），则调用 excute_cgi 函数执行 cgi 脚本*/
        else
        {
            execute_cgi(client, path, method, query_string);
        }
    }

    close(client);
}

```

t_mode 是用特征位来表示文件类型的，特征位的定义如下：

|  名称   |   值    | 描述                   |
| :-----: | :-----: | :--------------------- |
| S_IFMT  | 0170000 | 文件类型的位掩码       |
| S_IXUSR |  00100  | 文件所有者具可执行权限 |
| S_IFDIR | 0040000 | 目录                   |
| S_IXGRP |  00010  | 用户组具可执行权限     |
| S_IXOTH |  00001  | 其他用户具可执行权限   |


## cgi

```
/**********************************************************************/
/* Execute a CGI script.  Will need to set environment variables as
 * appropriate.
 * Parameters: client socket descriptor
 *             path to the CGI script */
/**********************************************************************/
void execute_cgi(int client, const char *path, const char *method, const char *query_string)
{
    char buf[1024];
    int cgi_output[2];
    int cgi_input[2];
    pid_t pid;
    int status;
    int i;
    char c;
    int numchars = 1;
    int content_length = -1;

    buf[0] = 'A';
    buf[1] = '\0';

    if (strcasecmp(method, "GET") == 0)
    {
        while ((numchars > 0) && strcmp("\n", buf)) /* read & discard headers */
            numchars = get_line(client, buf, sizeof(buf));
    }
    else if (strcasecmp(method, "POST") == 0) /*POST*/
    {
        numchars = get_line(client, buf, sizeof(buf));

        while ((numchars > 0) && strcmp("\n", buf))
        {
            buf[15] = '\0';

            if (strcasecmp(buf, "Content-Length:") == 0)
                content_length = atoi(&(buf[16]));

            numchars = get_line(client, buf, sizeof(buf));
        }
        if (content_length == -1)
        {
            bad_request(client);
            return;
        }
    }
    else /*HEAD or other*/
    {
        printf("execute_cgi >> HEAD or othe\n");
    }

	/**建立两个管道，cgi_input 和 cgi_output, 并 fork 一个进程*/
    if (pipe(cgi_output) < 0)
    {
        cannot_execute(client);
        return;
    }
    if (pipe(cgi_input) < 0)
    {
        cannot_execute(client);
        return;
    }

    if ((pid = fork()) < 0)
    {
        cannot_execute(client);
        return;
    }

    sprintf(buf, "HTTP/1.0 200 OK\r\n");
    send(client, buf, strlen(buf), 0);

	/**在子进程中，把 STDOUT 重定向到 cgi_outputt 的写入端，把 STDIN 重定向到 cgi_input 的读取端，
	关闭 cgi_input 的写入端 和 cgi_output 的读取端，设置 request_method 的环境变量，
	GET 的话设置 query_string 的环境变量，POST 的话设置 content_length 的环境变量，
	这些环境变量都是为了给 cgi 脚本调用，接着用 execl 运行 cgi 程序*/
    if (pid == 0) /* child: CGI script */
    {
        char meth_env[255];
        char query_env[255];
        char length_env[255];

        dup2(cgi_output[1], STDOUT);
        dup2(cgi_input[0], STDIN);
        close(cgi_output[0]);
        close(cgi_input[1]);
        sprintf(meth_env, "REQUEST_METHOD=%s", method);
        putenv(meth_env);

        if (strcasecmp(method, "GET") == 0)
        {
            sprintf(query_env, "QUERY_STRING=%s", query_string);
            putenv(query_env);
        }
        else /* POST */
        {
            sprintf(length_env, "CONTENT_LENGTH=%d", content_length);
            putenv(length_env);
        }

        execl(path, NULL);

        exit(0);
    }
	/**在父进程中，关闭 cgi_input 的读取端 和 cgi_output 的写入端，如果 POST 的话，
	把 POST 数据写入 cgi_input，已被重定向到 STDIN，
	读取 cgi_output 的管道输出到客户端，该管道输入是 STDOUT。
	接着关闭所有管道，等待子进程结束*/
    else /* parent */
    {
        close(cgi_output[1]);
        close(cgi_input[0]);

        if (strcasecmp(method, "POST") == 0)
        {
            for (i = 0; i < content_length; i++)
            {
                recv(client, &c, 1, 0);
                write(cgi_input[1], &c, 1);
            }
        }

        while (read(cgi_output[0], &c, 1) > 0)
            send(client, &c, 1, 0);

        close(cgi_output[0]);
        close(cgi_input[1]);

        waitpid(pid, &status, 0);
    }
}

```


**无名管道特点：**

1. 半双工，数据在同一时刻只能在一个方向上流动；
2. 数据只能从管道的一端写入，从另一端读出；
3. 写入管道的数据遵循先入先从出的规则；
4. 管道所传送的数据不是无格式的，这要求管道的读出方和写入方必须约定好数据的格式，如多少字节算一个消息；
5. 管道不是普通的文件，不属于某个文件系统，其只存在于内存中；
6. 管道在内存中对应一个缓冲区，不同的系统其大小不一定相同；
7. 从管道读数据是一次性操作，数据一旦被读走，它就从管道中丢弃，释放空间以便写更多的数据；
8. 管道没有名字，只能在具有公共祖先的进程(父进程和子进程，或两个兄弟进程)之间使用。



# 服务器与CGI间通信

**请求相关的环境变量**

|     string      | 描述                              |
| :-------------: | :-------------------------------- |
| REQUEST_METHOD  | 服务器与CGI程序之间的信息传输方式 |
|  QUERY_STRING   | 采用GET时所传输的信息             |
| CONTENT_LENGTH  | STDIO中的有效信息长度             |
|  CONTENT_TYPE   | 指示所传来的信息的MIME类型        |
|    PATH_INFO    | 路径信息                          |
| PATH_TRAMSLATED | CGI程序的完整路径名               |
|   SCRIPT_NAME   | 所调用的CGI程序名字               |

**服务器相关的环境变量**

|      string       | 描述                                  |
| :---------------: | :------------------------------------ |
| GATEWAY_INTERFACE | 服务器所实现的CGI版本                 |
|    SERVER_NAME    | 服务器的IP或名字                      |
|    SERVER_PORT    | 主机的端口号                          |
|  SERVER_SOFTWARE  | 调用CGI程序的HTTP服务器的名称和版本号 |

**客户端相关的环境变量**

|     string      | 描述                          |
| :-------------: | :---------------------------- |
|   REMOTE_ADDR   | 客户机的主机名                |
|   REMOTE_HOST   | 客户机的IP地址                |
|     ACCEPT      | 列出能被次请求接受的应答方式  |
| ACCEPT_ENCODING | 列出客户机支持的编码方式      |
| ACCEPT_LANGUAGE | 表明客户机可接受语言的IOS代码 |


## method

**POST**

客户端传入的用户数据将存放在 CGI 进程的标准输入中，同时将用户数据的长度赋予环境变量中的 CONTENT_LENGTH ，客户端用 POST 方式发送数据会有一个相应的 MIME 类型 (通用Internet邮件扩充服务)

目前，MIME类型一般是：application/x-wwww-form-urlencoded，该类型表示数据来自HTML表单。该类型记录在环境变量 CONTENT_TYPE 中，CGI程序应该检查该变量的值。

**GET**

CGI 无法直接充服务器的标准输入中获取数据，因为服务器把它从标准输入接收到的数据编码到环境变量 QUERY_STRING 中。


环境变量是一个保存用户信息的内存区。当客户端的用户通过浏览器发出 CGI 请求时，服务器就寻找本地的相应 CGI 程序并执行它。在执行 CGI 程序的同时，服务器把该用户的信息保存到环境变量里。

**CGI程序**

接下来，CGI 程序的执行流程是这样的：查询与该 CGI 程序进程相应的环境变量：第一步是 request_method ，如果是 POST，就获取环境变量的 CONTENT_LENGTH ，CONTENT_LENGTH 由服务器应用写入：

```
            sprintf(length_env, "CONTENT_LENGTH=%d", content_length);
            putenv(length_env);
```

然后 CGI 进程从相应的标准输入取出 CONTENT_LENGTH 长的数据。如果是GET，则用户数据就在环境变量的 QUERY_STRING 里。




# url

在HTTP协议消息头中，使用 Content-Type 来表示媒体类型信息。它被用来告诉服务端如何处理请求的数据，以及告诉客户端（一般是浏览器）如何解析响应的数据，比如显示图片，解析html或仅仅展示一个文本等。

Post请求的内容放置在请求体中，Content-Type定义了请求体的编码格式。数据发送出去后，还需要接收端解析才可以。接收端依靠请求头中的 Content-Type 字段来获知请求体的编码格式，最后再进行解析。


**url 编码的基本规则是：**

- 变量之间用“&”分开；
- 变量与其对应值用“=”连接；
- 空格用“+”代替；
- 保留的控制字符则用“%”连接对应的16禁止ASCII码代替；
- 某些具有特殊意义的字符也用“%”接对应的16进制ASCII码代替；
- 空格是非法字符；
- 任意不可打印的ASCII控制字符均为非法字符。

CGI 程序从标准输入或环境变量中获取客户端数据后，还需要进行解码。解码的过程就是URL编码的逆变：根据“&”和“=”分离HTML表单变量，以及特殊字符的替换。





CGI的格式输出内容必须组织成标题/内容的形式。CGI 标准规定了 CGI 程序可以使用的三个 HTTP 标题,标题必须占据第一行输出，而且必须随后带有一个空行


```
    sprintf(buf, "Content-type: text/html\r\n");
    send(client, buf, strlen(buf), 0);
```

|    string    | 描述                             |
| :----------: | :------------------------------- |
| Content-type | 设定随后输出数据所用的 MIME 类型 |
|   Location   | 设定输出为另外一个文档(url)      |
|    Status    | 指定 HTTP 状态码                 |



# web

我们来做 HTML 一个简单的页面来实现 CGI 的交互。












```
int get_inputs(void)
{

    int length;
    char *method;
    char *inputstring;

    /**将返回结果赋予指针*/
    method = getenv(“REQUEST_METHOD”);

    /**找不到环境变量REQUEST_METHOD，结束*/
    if (method == NULL)
        return 1;

    if (!strcasecmp(method, ”POST”)) // POST 方法
    {
        length = atoi(getenv(“CONTENT_LENGTH”));

        if (length != 0)
        {
            inputstring = malloc(sizeof(char) * length + 1); // 必须申请缓存，因为 stdin 是不带缓存的
            fread(inputstring, sizeof(char), length, stdin); // 从标准输入读取一定数据
        }		
    }
    else if (!strcasecmp(method, “GET”)) // GET 方法
    {
        Inputstring = getenv(“QUERY_STRING”);
        length = strlen(inputstring);
    }
    else
    {
    }

    if (length == 0)
        return 0;
}
```

