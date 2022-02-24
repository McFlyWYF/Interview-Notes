
# 53 - I. 在排序数组中查找数字 I

* 统计一个数字在排序数组中出现的次数。

* 输入: nums = [5,7,7,8,8,10], target = 8
* 输出: 2

#### 二分法


```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        count = 0
        while left < right: ############## 
            mid = (left + right) / 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] >= target: #############
                right = mid ##############

        while(left < len(nums) and nums[left] == target):
            count += 1
            left += 1
  
        return count
```
