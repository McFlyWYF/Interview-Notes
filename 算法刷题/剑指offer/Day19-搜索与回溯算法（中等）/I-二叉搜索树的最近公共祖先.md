
# 68 - I. 二叉搜索树的最近公共祖先

* 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
    * 最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

* 例如：
    * 输入: `root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8`
    * 输出: `6 `
    * 解释: 节点 2 和节点 8 的最近公共祖先是 6。

#### 方法一：递归法


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # p, q的值大于root，在右子树中找
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        # p, q的值小于root，在左子树中找
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        # 否则返回root
        return root
```

* 时间复杂度：$O(N)$
* 空间复杂度：$O(N)$

#### 方法二：迭代法


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        while root:
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                break
        return root
```

* 时间复杂度：$O(N)$
* 空间复杂度：$O(1)$
