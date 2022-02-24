
# 55 - I. 二叉树的深度

* 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

* 例如：
    * 给定二叉树 [3,9,20,null,null,15,7]，
    * 返回它的最大深度 3 。

#### 解题思路

##### 方法一：递归（后序遍历）


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

##### 方法二：迭代法（层序遍历）


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        from collections import deque
        
        if root == None:
            return 0
        
        maxdepth = 0
        q = deque([root])
        while q:
            size = len(q)
            maxdepth += 1
            for i in range(size):
                cur = q.popleft()
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return maxdepth
```
