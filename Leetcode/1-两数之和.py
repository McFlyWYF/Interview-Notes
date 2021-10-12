# 暴力枚举
'''
两层循环，外层循环和其余元素相加判断是否等于target
'''

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        index = []

        for i in range(1, len(nums)):
            for j in range(i, len(nums)):
                if(nums[i - 1] + nums[j] == target):
                    index.append(i - 1)
                    index.append(j)
        return index


# 哈希表
'''
将原来的元素存储到哈希表中，如果存在target-value的某个元素在哈希表中，则返回下标
'''
class Solution1(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for index, value in enumerate(nums):
            anotherValue = target - value
            if anotherValue in hashmap:
                return [hashmap[anotherValue], index]
            hashmap[value] = index
        return None

s = Solution1()
nums = [1,2,3,4]
index = s.twoSum(nums, target=5)
print(index)