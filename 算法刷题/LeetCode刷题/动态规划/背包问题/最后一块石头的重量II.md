
# 最后一块石头的重量II

### 有一堆石头，每块石头的重量都是正整数。每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为x和y，且x<=y。那么粉碎的可能结果如下：

* 如果x==y，那么两块石头都会被完全粉碎。
* 如果x!=y，那么重量为x的石头将会完全粉碎，而重量为y的石头新重量为y-x。

### 最后，最多只会剩下一块石头。返回此石头最小的可能重量。如果没有石头剩下，就返回0.

* 例如：
    * 输入：[2,7,4,1,8,1]
    * 输出：1

* 尽量让石头分成重量相同的两堆，相撞之后剩下的石头最小，这样就转化为01背包问题。
    * 石头重量=物品重量
    * 石头重量=物品价值
    * 石头重量的一半=背包容量
    * 每次取一个


```python
def solve(nums):
    if sum(nums) % 2 == 0:
        bagweight = sum(nums) // 2
    else:
        bagweight = sum(nums) // 2 + 1
    print(bagweight)
    # 定义，初始化    
    dp = [0 for _ in range(bagweight + 1)]
    # 遍历
    for i in range(len(nums)):
        for j in range(bagweight, nums[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
            
    print(dp)
    return abs(sum(nums) - dp[-1] - dp[-1])  # 用总重量减去背包最大重量等于另一堆的重量，再两者相减，得到差
```


```python
nums = [31,26,33,21,40]
solve(nums)
```

    76
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 21, 21, 26, 26, 26, 26, 26, 31, 31, 33, 33, 33, 33, 33, 33, 33, 40, 40, 40, 40, 40, 40, 40, 47, 47, 47, 47, 47, 52, 52, 54, 54, 54, 57, 57, 59, 59, 61, 61, 61, 64, 64, 66, 66, 66, 66, 66, 71, 71, 73, 73, 73, 73]
    




    5



* 时间复杂度：$O(m*n)$，m是石头总重量，n是石头块数
* 空间复杂度：$O(m)$
