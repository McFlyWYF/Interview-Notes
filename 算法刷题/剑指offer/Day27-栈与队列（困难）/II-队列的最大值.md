
# 59 - II. 队列的最大值

* 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。若队列为空，pop_front 和 max_value 需要返回 -1

* 例如：
    * 输入: `["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]`
   ` [[],[1],[2],[],[],[]]`
    * 输出: `[null,null,null,2,1,2]`

### 解题思路

* 函数设计：
    * 初始化队列 queue（保存入栈元素），双向队列 deque（保存最大值）；

* 最大值 max_value() ：
    * 当双向队列 deque 为空，则返回 -1；
    * 否则，返回 deque 队首元素；

* 入队 push_back() ：
    * 将元素 value 入队 queue ；
    * 将双向队列中队尾所有小于 value 的元素弹出（以保持 deque 非单调递减），并将元素 value 入队 deque ；

* 出队 pop_front() ：
    * 若队列 queue 为空，则直接返回 -1；
    * 否则，将 queue 首元素出队；
    * 若 deque 首元素和 queue 首元素相等 ，则将 deque 首元素出队（以保持两队列元素一致 ） ；


```python
import queue

class MaxQueue:

    def __init__(self):
        self.deque1 = queue.Queue()
        self.deque2 = queue.deque()

    def max_value(self) -> int:
        return self.deque2[0] if self.deque2 else -1


    def push_back(self, value: int) -> None:
        self.deque1.put(value)
        while self.deque2 and self.deque2[-1] < value:
            self.deque2.pop()
        self.deque2.append(value)

    def pop_front(self) -> int:
        if self.deque1.empty():
            return -1
        
        tmp = self.deque1.get()
        if tmp == self.deque2[0]:
            self.deque2.popleft()
        return tmp

# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```

* 时间复杂度：$O(1)$
* 空间复杂度：$O(n)$
