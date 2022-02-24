
# 56 - II. 数组中数字出现的次数 II

* 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

* 例如：
    * 输入：`nums = [3,4,3,3]`
    * 输出：4

### 解题思路

#### 方法1:哈希表


```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = {}
        for num in nums:
            d[num] = 1 + d.get(num, 0)
        print(d)
        for (key, value) in d.items():
            if value == 1:
                return key
```


```python
s = Solution()
s.singleNumber([1, 4, 4, 4])
```

    {1: 1, 4: 3}
    




    1



* 时间复杂度：$O(N)$
* 空间复杂度：$O(N)$
