
# 买卖股票的最佳时机III

### 给定一个数组，它的第i个元素是一支给定的股票在第i天的价格。

* 设计一个算法来计算所能获取的最大利润。`最多`可以完成两笔交易。

* 例如：
    * 输入：prices=[3,3,5,0,0,3,1,4]
    * 输出：6

##### 1.确定dp数组
* 一天一共有5种状态：
    * 0：没有操作
    * 第一次买入
    * 第一次卖出
    * 第二次买入
    * 第二次卖出
    * `dp[i][j]`中的i表示第i天，j为[0-4]五种状态，dp[i][j]表示第i天状态j所剩最大现金。
    
##### 2.确定递推公式
* dp[i][0]表示没有操作
    * `dp[i][0] = dp[i - 1][0]`
* dp[i][1]
    * 第i天买入股票了，`dp[i][1] = dp[i - 1][0] - prices[i]`
    * 第i天没有操作，`dp[i][1] = dp[i - 1][1]`
    * `dp[i][1] = max(dp[i - 1][0] - prices[i], dp[i - 1][1])`
* dp[i][2]
    * 第i天卖出股票了，`dp[i][2] = prices[i] + dp[i - 1][1]`
    * 第i天没有操作，`dp[i][2] = dp[i - 1][2]`
    * `dp[i][2] = max(prices[i] + dp[i - 1][1], dp[i - 1][2])`
* dp[i][3]
    * `dp[i][3] = max(dp[i - 1][2] - prices[i], dp[i - 1][3])`
* dp[i][4]
    * `dp[i][4] = max(prices[i] + dp[i - 1][3], dp[i - 1][4])`
    
##### 3.初始化
* `dp[0][0] = 0`
* `dp[0][1] = -prices[0]`
* `dp[0][2] = 0`
* `dp[0][3] = -prices[0]`
* `dp[0][4] = 0`


```python
def solve(prices):
    if len(prices) == 0:
        return 0
    # 定义dp数组
    dp = [[0] * 5 for _ in range(len(prices))]
    # 初始化
    dp[0][1] = -prices[0]
    dp[0][3] = -prices[0]
    # 遍历
    for i in range(1, len(prices)):
        dp[i][0] = dp[i-1][0]
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i])
        dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i])
        dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i])
    print(dp)
    return dp[-1][4]
```


```python
prices = [3,3,5,0,0,3,1,4]
solve(prices)
```

    [[0, -3, 0, -3, 0], [0, -3, 0, -3, 0], [0, -3, 2, -3, 2], [0, 0, 2, 2, 2], [0, 0, 2, 2, 2], [0, 0, 3, 2, 5], [0, 0, 3, 2, 5], [0, 0, 4, 2, 6]]
    




    6



* 时间复杂度：$O(n)$
* 空间复杂度：$O(n*5)$
