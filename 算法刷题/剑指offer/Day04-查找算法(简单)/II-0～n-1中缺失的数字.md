
# 53 - II. 0～n-1中缺失的数字

* 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

* 输入: [0,1,3]
* 输出: 2

#### 暴力解法


```python
class Solution(object):
    def missingNumber(self, nums):
        for i in range(len(nums)):
            if nums[i] != i:
                return i
        return len(nums)
```

#### 二分法


```python
class Solution(object):
    def missingNumber(self, nums):
        # 二分查找
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) / 2
            if nums[mid] == mid: # 如果相等，说明缺失的在后半区间
                left = mid + 1
            else:  # 如果不相等，缺失的在前半区间
                right = mid - 1

        return left
```
