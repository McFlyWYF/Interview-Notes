
# 56 - I. 数组中数字出现的次数

* 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

* 例如：
    * 输入：`nums = [4,1,4,6]`
    * 输出：`[1,6] 或 [6,1]`

#### 解题思路

* 根据异或操作，相同取0，不同则相加。

* 遍历nums执行异或：
![image.png](attachment:21bea9e1-ec3b-4910-899b-188639218983.png)

* 循环左移计算m：
![image.png](attachment:b38f0dc7-9afa-4114-94ad-8740782f57c5.png)

* 拆分nums为两个子数组；

* 分别遍历两个子数组执行异或：
    * 通过遍历判断 nums 中各数字和 m 做与运算的结果，可将数组拆分为两个子数组，并分别对两个子数组遍历求异或，则可得到两个只出现一次的数字。

* 返回值：
    * 返回只出现一次的数字x,y即可。


```python
class Solution(object):
    def singleNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        x, y, n, m = 0, 0, 0, 1
        for num in nums:         # 1. 遍历异或
            n ^= num
        while n & m == 0:        # 2. 循环左移，计算 m
            m <<= 1       
        for num in nums:         # 3. 遍历 nums 分组
            if num & m: x ^= num # 4. 当 num & m != 0
            else: y ^= num       # 4. 当 num & m == 0
        return x, y              # 5. 返回出现一次的数字
```


```python
s = Solution()
s.singleNumbers([4, 1, 4, 6])
```




    (1, 6)



* 时间复杂度：$O(N)$
* 空间复杂度：$O(1)$
