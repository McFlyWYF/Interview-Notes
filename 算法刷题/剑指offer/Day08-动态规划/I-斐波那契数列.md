
# 10- I. 斐波那契数列


```python
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        # 定义
        dp = [0 for i in range(n + 1)]
        # 初始化
        dp[0] = 0
        dp[1] = 1
        # 遍历
        for i in range(2, n + 1):
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007
        return dp[-1]
```


```python
s = Solution()
n = 45
s.fib(n)
```




    134903163


