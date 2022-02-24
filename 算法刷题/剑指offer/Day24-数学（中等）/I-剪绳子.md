
# 14- I. 剪绳子

* 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m-1]` 。请问` k[0]*k[1]*...*k[m-1]` 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

* 例如：
    * 输入：10
    * 输出：36
    * 解释：10 = 3 + 3 + 4，3 x 3 x 4 = 36


```python
class Solution(object):
    def cuttingRope(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 3:
            return n - 1
        b = n % 3
        if b == 0:
            return pow(3, n // 3)
        elif b == 1:
            return pow(3, n // 3 - 1) * 4
        elif b == 2:
            return pow(3, n // 3) * 2
```

* 时间复杂度：$O(1)$
* 空间复杂度：$O(1)$
