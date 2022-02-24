
# 买卖股票的最佳时机IV

### 给定一个整数数组prices，它的第i个元素prices[i]是一支给定的股票在第i天的价格。

* 设计一个算法来计算你所能获取的最大利润。你最多可以完成k笔交易。

* 例如：
    * 输入：k=2, prices=[2,4,1]
    * 输出：2

##### 定义dp数组
* dp[i][j]：第i天的状态为j，所剩下的最大现金是dp[i][j]。

##### 确定递推公式
* dp[i][0]：没有操作
* dp[i][1]：买入，dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
* dp[i][2]：卖出，dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
* ......

##### 初始化
* dp[0][0] = 0
* dp[0][1] = -prices[0]
* dp[0][2] = 0


```python
def solve(k, prices):
    if len(prices) == 0 or len(prices) == 1:
        return 0
    # 定义
    dp = [[0 for _ in range(k * 2 + 1)] for _ in range(len(prices))]
    # 初始化
    for i in range(1, 2 * k + 1, 2):
        dp[0][i] = -prices[0]
    # 遍历
    for i in range(1, len(prices)):
        for j in range(0, 2 * k - 1, 2):
            dp[i][j + 1] = max(dp[i-1][j + 1], dp[i-1][j] - prices[i])
            dp[i][j + 2] = max(dp[i-1][j + 2], dp[i-1][j + 1] + prices[i])
    
    print(dp)
    return dp[-1][2 * k]
```


```python
k = 2
prices = [3,2,6,5,0,3]
solve(k, prices)
```

    [[0, -3, 0, -3, 0], [0, -2, 0, -2, 0], [0, -2, 4, -2, 4], [0, -2, 4, -1, 4], [0, 0, 4, 4, 4], [0, 0, 4, 4, 7]]
    




    7



和买卖股票的最佳时机III类似，只不过最多两次变为k次。
