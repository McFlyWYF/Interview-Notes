
# 10- II. 青蛙跳台阶问题

* 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

* 例如:
    * 输入：n = 2
    * 输出：2

* 找规律,除了n=0,1之外,其余都是前两个之和.`dp[i] = dp[i - 1] + dp[i - 2]`.


```python
class Solution(object):
    def numWays(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 1
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007
        print(dp)
        return dp[-1]
```


```python
s = Solution()
n = 5
s.numWays(n)
```

    [1, 1, 2, 3, 5, 8]
    




    8


