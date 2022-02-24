
# 买卖股票的最佳时机II

### 给定一个数组，它的第i个元素是一支给定股票第i天的价格。

* 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易。

* 例如：
    * 输入：[7,1,5,3,6,4]
    * 输出：7

* dp[i][0]：第i天持有股票所得现金
    * 第i-1天持有股票`dp[i - 1][0]`
    * 第i天买入股票，所得现金是昨天不持有股票所得现金减去今天的股票价格`dp[i-1][1]-prices[i]`，这里是因为股票可以多次买卖，
* dp[i][1]：第i天不持有股票所得现金
    * 第i-1天不持有`dp[i - 1][1]`
    * 第i-1天持有，第i天卖掉，`prices[i] + dp[i-1][0]`


```python
def solve(prices):
    dp = [[0 for _ in range(2)] for _ in range(len(prices))]
    dp[0][0] = -prices[0]
    dp[0][1] = 0
    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i - 1][0],dp[i - 1][1] -prices[i])  # 区别
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
    print(dp)
    return dp[-1][1]
```


```python
prices = [7,1,5,3,6,4]
solve(prices)
```

    [[-7, 0], [-1, 0], [-1, 4], [1, 4], [1, 7], [3, 7]]
    




    7



* 时间复杂度：$O(n)$
* 空间复杂度：$O(n)$

#### 贪心算法
* 收集每天的正利润即可
