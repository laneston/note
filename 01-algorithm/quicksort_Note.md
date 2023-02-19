## 快速排序的基本思想：

通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据标称有序序列。

这个跟二分法的思想类似，是一种分而治之思想。

## 应用讲解

有一个数组data，当中包含10个数据：2，8，3，9，6，7，0，4，1，5

我们现在要对它们进行一个快速排序。排序之前我们需要找一个参照物，这个参照物是随机选择的。我们可以选第一个数字，也可以选最后一个，当然也能选中间一个数字。在这里，我们选择第一个数字当做参照物(Basic)，即 Basic = data[Firstdata]。我们还需设定两个指向除参照物外的左右两端元素的指针，在这里左指针 Lpointer 指向 data[1]，右指针 Rpointer 指向 data[9]。在这里，左指针 Lpointer 用来找出比 Basic 小的元素，右指针 Rpointer 找出比参照物 Basic 大的元素。

|2|8|3|9|6|7|0|4|1|5|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Basic|Lpointer|-|-|-|-|-|-|-|Rpointer|

**执行步骤：**

0. 设第一个元素为 Basic；
1. Rpointer 向左移动，然后分别比较指向的元素是否大于 Basic ，直至遇到比 Basic 小时停止移动，本例子中 Rpointer 在指向元素 5 时就停止移动；
2. 左指针向右移动，然后分别比较指向的元素是否小于 Basic ，直至遇到比 Basic 大时停止移动，这里 Lpointer 到达元素 0 时停止移动；
3. 此时开始进行数据交换， Lpoint 与 Rpoint 中的元素交换位置。这时数据顺序变为：

|2|8|3|9|6|7|5|4|1|0|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Basic|-|-|-|-|-|Lpointer|-|-|Rpointer|

4. 右指针继续移动，在到达4时停了下来；
5. 左指针继续移动，到达4时与右指针相遇，此时也需要停下来；
6. 当左右指针停止移动，且在同一位置时，将当前指向的元素与 Basic 交换，然后 Basic 的位置就成功放目标位置上。即：

|4|8|3|9|6|7|5|**2**|1|0|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|-|-|-|-|-|-|-|Basic|-|-|

7. 第一轮排序已经结束，我们此时进行第二轮排序。将 Basic 左侧和右侧的元素重复 0 1 2 3 4 5 6 步骤。

情况依次变为：

4，8，5，9，6，7，3，**2**，1，0

7，8，5，9，6，**4**，3，**2**，**1**，**0**

7，8，9，5，6，**4**，3，**2**，**1**，**0**

**9**，**8**，**7**，5，6，**4**，**3**，**2**，**1**，**0**

**9**，**8**，**7**，**6**，**5**，**4**，**3**，**2**，**1**，**0**

到现在为止我们就已经将这10个元素按照倒序排列完毕。

如果想要按正序排列，只需修改1,2点条件即可。这时有朋友可能会问了，为什么是右指针向左移动先执行。在快速排序中，如果 Basic 左边的元素多于右边，即左指针先执行移动，如果 Basic 右边的元素多于左边，即右指针先执行移动。有闲情逸致的朋友可以试试看如果不按照这个规律排序会出现什么问题。

## 代码实现

我们开始编写一个 C 代码来实现快速排序。

```
/**
  * @brief  quick sort function
  * @param  data: the buff of the data
  *         FirstData: the footcode of first data that need to be sort in the arry
  *         LastData:  the footcode of last data that need to be sort in the arry
  * @retval None
  */
void quicksort(short int *data, short int FirstData, short int LastData)
{    
    short int Lpointer = FirstData;
    short int Rpointer = LastData;
    /*the first data is Basic*/
    short int Basic = data[Lpointer];

    /*If two pointers meet, end the round.*/
    if (Lpointer >= Rpointer)
    {
        return ;
    }    

    while (Lpointer < Rpointer)
    {
        /* If Lpointer and Rpointer had not met, and the data pointed by Rpointer less than Basic. Rpointer move to left*/
        while ((Lpointer < Rpointer) && (data[Rpointer] >= Basic))
        {
            Rpointer--;
        }
        data[Lpointer] = data[Rpointer];/*Put the data[Rpointer] to data[Lpointer]*/

        /* If Lpointer and Rpointer had not met, and the data pointed by Rpointer geater than Basic. Lpointer move to right*/
        while ((Lpointer < Rpointer) && (data[Lpointer] < Basic))
        {
            Lpointer++;
        }
        data[Rpointer] = data[Lpointer]; /*Put the data[Lpointer] to data[Rpointer]*/
    }

    data[Lpointer] = Basic; /*Put the Basic to data[Lpointer]*/

    quicksort(data, FirstData, Lpointer-1);
    quicksort(data, Lpointer+1, Rpointer);    
}
```
