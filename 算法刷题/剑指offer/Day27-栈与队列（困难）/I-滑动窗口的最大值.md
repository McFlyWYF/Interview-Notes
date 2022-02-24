
# 59 - I. 滑动窗口的最大值

* 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

* 例如：
    * 输入: `nums = [1,3,-1,-3,5,3,6,7], 和 k = 3`
    * 输出: `[3,3,5,5,6,7] `

### 解题思路

* 初始化： 双端队列 deque，结果列表 res，数组长度 n；
* 滑动窗口： 左边界范围 $i \in [1 - k, n - k]$ ，右边界范围 $j \in [0, n - 1]$；
    * 若 i > 0 且 队首元素 deque[0] = 被删除元素 nums[i - 1] ：则队首元素出队；
    * 删除 deque 内所有 < nums[j] 的元素，以保持 deque 递减；
    * 将 nums[j] 添加至 deque 尾部；
    * 若已形成窗口（即 $i \geq 0$）：将窗口最大值（即队首元素 deque[0] ）添加至列表 res ；
* 返回值： 返回结果列表 res；


```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        deque = collections.deque()
        res = []
        n = len(nums)
        for i, j in zip(range(1 - k, n + 1 - k), range(n)):
            # 删除deque中对应的nums[i - 1]
            if i > 0 and deque[0] == nums[i - 1]:
                deque.popleft()
            # 保持deque递减
            while deque and deque[-1] < nums[j]:
                deque.pop()

            deque.append(nums[j])
            # 记录窗口最大值
            if i >= 0:
                res.append(deque[0])
        return res
```

* 时间复杂度：$O(n)$
* 空间复杂度：$O(k)$


```python
a = [1,2,3]
a.pop()
```




    3


