
# 57 - II. 和为s的连续正数序列

* 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

* 例如：
    * 输入：`target = 9`
    * 输出：`[[2,3,4],[4,5]]`

#### 解题思路（双指针）

* 定义left和right指针，初始化`left=1,right=2,sums=3`；
* 如果`sums == target`: 将结果保存到res中，left指针加1，`sums -= left`；
* 如果`sums < target: right += 1, sums += right`；
* 如果`sums > target: sums -= left, left += 1`；

![image.png](attachment:b4fc89ab-fec3-4506-8c50-7f3e7e781811.png)


```python
class Solution(object):
    def findContinuousSequence(self, target):
        """
        :type target: int
        :rtype: List[List[int]]
        """
        left = 1
        right = 2
        sums = 3
        res = []
        while left < target:
            if sums == target:
                res.append(list(range(left, right + 1)))
                sums -= left
                left += 1
            elif sums > target:
                sums -= left
                left += 1
            else:
                right += 1
                sums += right
        return res
```

* 时间复杂度：$O(N)$
* 空间复杂度：$O(1)$
