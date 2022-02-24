
# 30. 包含min函数的栈

* 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```


```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.s1 = list()
        self.s2 = list()

    def push(self, x: int) -> None:
        self.s1.append(x)
        if not self.s2 or self.s2[-1] >= x:
            self.s2.append(x)
        
    def pop(self) -> None:
        if self.s1.pop() == self.s2[-1]:
            self.s2.pop()

    def top(self) -> int:
        return self.s1[-1]

    def min(self) -> int:
        return self.s2[-1]
```
