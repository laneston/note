
```
#include <sys/epoll.h>
```

int epoll_create1(int flags);

功能：创建一个多路复用的实例

参数：

flags：

0：如果这个参数是0，这个函数等价于poll_create（0）

EPOLL_CLOEXEC：这是这个参数唯一的有效值，如果这个参数设置为这个。那么当进程替换映像的时候会关闭这个文件描述符，这样新的映像中就无法对这个文件描述符操作，适用于多进程编程 + 映像替换的环境里

返回值：

- success：返回一个非0 的未使用过的最小的文件描述符
- error：-1 errno被设置

--------------------------------------------------------------------

int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);

功能：操作一个多路复用的文件描述符

参数：

epfd：epoll_create1的返回值

op：要执行的命令

- EPOLL_CTL_ADD：向多路复用实例加入一个连接socket的文件描述符；
- EPOLL_CTL_MOD：改变多路复用实例中的一个socket的文件描述符的触发事件；
- EPOLL_CTL_DEL：移除多路复用实例中的一个socket的文件描述符。

fd：要操作的socket的文件描述符

event：

```
typedef union epoll_data {
               void        *ptr;
               int          fd;
               uint32_t     u32;
               uint64_t     u64;
           } epoll_data_t;

struct epoll_event {
               uint32_t     events;      /* Epoll events */
               epoll_data_t data;        /* User data variable */
};
```

events 可以是下列命令的任意按位与:
1. EPOLLIN：对应的文件描述有可以读取的内容
2. EPOLLOUT：对应的文件描述符有可以写入
3. EPOLLRDHUP：写到一半的时候连接断开
4. EPOLLPRI：发生异常情况，比如所tcp连接中收到了带外消息
5. EPOLLET：设置多路复用实例的文件描述符的事件触发机制为边沿触发，默认为水平触发

1、当多路复用的实例中注册了一个管道，并且设置了触发事件EPOLLIN，

2、管道对端的写入2kb的数据，

3、epoll_wait收到了一个可读事件，并向上层抛出，这个文件描述符

4、调用者调用read读取了1kb的数据，

5、再次调用epoll_wait


边沿触发：上面的调用结束后，在输入缓存区中还有1kb的数据没有读取，但是epoll_wait将不会再抛出文件描述符。这就导致接受数据不全，对端得不到回应，可能会阻塞或者自己关闭
因为边沿触发的模式下，只有改变多路复用实例中某个文件描述符的状态，才会抛出事件。
相当于，边沿触发方式，内核只会在第一次通知调用者，不管对这个文件描述符做了怎么样的操作

水平触发：
只要文件描述符处于可操作状态，每次调用epoll_wait，内核都会通知你

- EPOLLONESHOT：epoll_wait只会对该文件描述符第一个到达的事件有反应，之后的其他事件都不向调用者抛出。需要调用epoll_ctl函数，对它的事件掩码重新设置
- EPOLLWAKEUP
- EPOLLEXCLUSIVE

返回值：

- success：0
- error：-1 errno被设置

--------------------------------------------------------------------

int epoll_wait(int epfd, struct epoll_event *events,int maxevents, int timeout);

功能：等待一个epoll队列中的文件描述符的I/O事件发生

参数：

- epfd：目标epoll队列的描述符；
- events：用于放置epoll队列中准备就绪（被触发）的事件；
- maxevents：最大事件数目；
- timeout：指定函数超时时间，在阻塞指定时间后解除阻塞。

返回值：

- \>=0，表示准备就绪的文件描述符个数;
- -1：出错，errno被设置。
