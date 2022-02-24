
# 零钱兑换II

### 给定不同面额的硬币和一个总金额，写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。

* 例如：
    * 输入：amount = 5，coins = [1, 2, 5]
    * 输出：4

* 确定dp数组
    * dp[j]：凑成总金额为j的组合数为dp[j]
* 确定递推公式
    * dp[j]就是所有的dp[j - coins[i]]相加
    * `dp[j] += dp[j - coins[i]]`
* dp数组初始化
    * dp[0]=1，凑成总金额为0的组合数为1，其他初始化为0
* 确定遍历顺序
    * 先遍历物品，再遍历背包，这样计算出来的是组合数
    * 先遍历背包，再遍历物品，计算出来的是排列数，包含了重复的集合，这种遍历顺序是不行的。


```python
def solve(amount, coins):
    dp = [0 for _ in range(amount + 1)]
    dp[0] = 1
    for i in range(len(coins)):
        for j in range(coins[i], amount + 1):
            dp[j] += dp[j - coins[i]]
    print(dp)
    return dp[-1]
```


```python
amount = 5
coins = [1, 2, 5]
solve(amount, coins)
```

    [1, 1, 2, 2, 3, 4]
    




    4



#### 遍历顺序

* 如果求组合数，先遍历物品，再遍历背包。
* 如果求排列数，先遍历背包，再遍历物品。
