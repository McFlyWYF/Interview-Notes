### 集合划分问题

##### Q：输入一个数组`nums`和一个正整数`k`，请你判断`nums`是否能够被平分为元素和相同的`k`个子集。

* 有两种不同的视角
  * 视角1：切换到n个数字的视角，每个数字都要选择进入到`k`个桶中的某一个。
  * 视角2：切换到k个桶的视角，对于每个桶，都要遍历`nums`中的`n`个数字，然后选择是否将当前遍历到的数字装进自己的桶里。

* 以数字的视角

  * 递归遍历数组

  ```python
  def traverse(nums, index):
  	if index == len(nums):
      	return
    traverse(nums, index + 1)
  ```

  * 解决方法

  ```python
  def canPartitionKSubsets(nums, k):
    if k > len(nums):
      return False
    # 不能平均分配
    if sum(nums) % k != 0:
      return False
    bucket = [0 for _ in range(k)]   # k个桶初始化
    target = sum(nums) // k  # 每个桶的数字和
    
    # 优化，从大到小排序，大的数字先分配到bucket中，对之后的数字，只会更大，更容易触发剪枝的if条件，减少递归调用
    nums.sort(reverse=True)
  
  	def backtrack(nums):
      # 所有的数字都已装进桶里
      if index == len(nums):
        return True
      for i in range(len(bucket)):
        # 剪枝
        if bucket[i] + nums[index] > target:
          continue
        # 装进桶
        bucket[i] += nums[index]
        if backtrack(index + 1):
          return True
        # 撤销操作
        bucket[i] -= nums[index]
      return False
    return backtrack(0)
  ```

* 以桶的视角（时间复杂度更低）

  ```python
  class Solution(object):
      def canPartitionKSubsets(self, nums, k):
          """
          :type nums: List[int]
          :type k: int
          :rtype: bool
          """
          # 方便剪枝
          nums.sort(reverse=True)
          if k > len(nums):return False
          if sum(nums) % k != 0:return False
          used = [False for _ in range(len(nums))]
          target = sum(nums) // k
  
          def backtrack(nums, k, bucket, start, used, target):
              # 所有桶装满了
              if k == 0:
                  return True
              if bucket == target:
                  # 该桶已装满，递归下一桶
                  return backtrack(nums, k - 1, 0, 0, used, target)
              for i in range(start, len(nums)):
                	# 数字i已被装进桶
                  if used[i]:
                      continue
                  if nums[i] + bucket > target:
                      continue
                  # 将数字i装进桶里
                  used[i] = True
                  bucket += nums[i]
                  # 递归下一个数字
                  if backtrack(nums, k, bucket, i + 1, used, target):
                      return True
                  # 撤销操作
                  used[i] = False
                  bucket -= nums[i]
              return False
          return backtrack(nums, k, 0, 0, used, target)
  ```

* 时间复杂度

  * 第一种：每个元素都有k种选择，所以是`O(k^n)`
  * 第二种：每个桶要遍历n个数字，选择`[装入]`或`[不装入]`，有`2^n`种结果，有k个桶，所以时间复杂度是`O(k*2^n)`
