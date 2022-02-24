
# 54. 二叉搜索树的第k大节点

* 给定一棵二叉搜索树，请找出其中第 k 大的节点的值。

* 例如：
    * 输入: `root = [3,1,4,null,2], k = 1`
    * 输出: `4`

#### 解题思路

* 中序遍历二叉树，因为其是有序的，输出倒数第k个元素即可。

##### 迭代法


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthLargest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        if not root:
            return []
        stack = []  # 不能提前将root结点加入stack中
        result = []
        cur = root
        while cur or stack:
            # 先迭代访问最底层的左子树结点
            if cur:     
                stack.append(cur)
                cur = cur.left
            # 到达最左结点后处理栈顶结点    
            else:
                cur = stack.pop()
                result.append(cur.val)
                # 取栈顶元素右结点
                cur = cur.right
        return result[-k]
```

##### 递归法


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthLargest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        def travel(cur, result):
            if cur == None:
                return
            travel(cur.left, result)
            result.append(cur.val)
            travel(cur.right, result)

        result = []
        travel(root, result)
        return result[-k]
```
