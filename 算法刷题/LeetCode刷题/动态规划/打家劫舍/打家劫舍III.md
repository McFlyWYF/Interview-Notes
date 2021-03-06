
# 打家劫舍III

### 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。除了“根”之外，每栋房子只有一个“父”房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

* 计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

* 例如：
    * 输入：[3,2,3,null,3,null,1]
    * 输出：7

* 每个节点的情况就是偷与不偷，其余和前两个打家劫舍一样。

* dp数组的含义：长度为2，下标为0记录偷该节点所得到的最大金钱，下标为1记录不偷该节点所得到的最大金钱。

* 遍历顺序：
    * 使用后序遍历
        * 递归左节点，得到左节点偷与不偷的金钱。
        * 递归右节点，得到右节点偷与不偷的金钱。


```python
def solve(root):

    def robTree(cur):
        # 初始化
        if cur == None:
            return (0, 0)
        left = robTree(cur.left)
        right = robTree(cur.right)
        
        # 下标0:偷，下标1:不偷
        # 偷当前节点
        val1 = cur.val + left[1] + right[1]
        # 不偷当前节点
        val2 = max(left[0], left[1]) + max(right[0], right[1])
        return (val1, val2)
    
    result = robTree(root)
    # 返回偷与不偷的最大值
    return max(result[0], result[1])
```

* 时间复杂度：$O(n)$，遍历了每个节点
* 空间复杂度：$O(n)$，包括递归系统栈的时间

第一次树的遍历和动态规划结合的题目。
