
# 57. 和为s的两个数字

* 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

* 例如：
    * 输入：`nums = [2,7,11,15], target = 9`
    * 输出：`[2,7] 或者 [7,2]`


```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        left = 0
        right = len(nums) - 1

        while left < right:
            if nums[left] + nums[right] > target:  # 两者相加大于target，右指针-1
                right -= 1
            elif nums[left] + nums[right] < target:  # 两者相加小于target，左指针+1
                left += 1
            else:
                return [nums[left], nums[right]] 
```


```python
s = Solution()
nums = [2,7,9,11]
s.twoSum(nums, 9)
```




    [2, 7]



* 时间复杂度：$O(n)$
* 空间复杂度：$O(1)$
