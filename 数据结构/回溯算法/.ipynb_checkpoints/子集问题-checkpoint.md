### 子集问题

##### Q1：输入一个不包含重复数字的数组`[1,2,3]`，要求输出这些数字的所有子集。

* 子集问题就是收集树形结构中树的所有节点的结果。

* 第一种解法是利用**数学归纳法**的思想，`求[1,2,3]的子集`，如果知道了[1,2]的子集，subset([1,2,3])等于subset([1,2])的结果中每个集合再添加上 3，如果`A=subset([1,2,3])`，那么`subset([1,2,3])=A+[A[i].add(3) for i = 1...len(A)]`。一个典型的递归结构。

* 第二种通用方法就是回溯算法。

  ![image-20210902155811225](/Users/wangyufei/Library/Application Support/typora-user-images/image-20210902155811225.png)

  终止条件就是集合为空的时候，就是叶子节点。

  ```python
  res = []
  path = []   # 子集
  def backtrack(nums, start):   # 这里可以不加终止条件，因为每次递归已经遍历到叶子节点了
    res.append(path[:])
    for i in range(start, len(nums)):
      path.append(nums[i])
      backtrack(nums, i + 1)
      path.pop()
  backtrack(nums, 0)
  return res
  ```

##### Q2：给你一个整数数组，可能包含重复元素，例如`[1,2,2]`，返回该数组所有可能的子集。

* 本题方法的重点就是去重。去重方法：对同一层使用过的元素进行跳过。

  ```python
  res = []
  path = []
  def backtrack(nums, start):
    # if path[:] not in res:   # 当前要添加的子集不存在于集合中
    res.append(path[:])
    for i in range(start, nums):
      if i > start and nums[i] == nums[i - 1]:   # 对同一层使用过的元素进行跳过
        continue
      path.append(nums[i])
      backtrack(nums, i + 1)
      path.pop()
      nums = nums.sort()
      backtrack(nums, 0)
      return res