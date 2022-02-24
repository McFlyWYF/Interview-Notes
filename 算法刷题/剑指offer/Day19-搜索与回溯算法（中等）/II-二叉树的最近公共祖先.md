
# 68 - II. 二叉树的最近公共祖先

* 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
    * 最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

* 例如：
    * 输入: `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1`
    * 输出: `3`
    * 解释: 节点 5 和节点 1 的最近公共祖先是节点 3。

#### 解题思路

* 若 root 是 p, q 的 最近公共祖先 ，则只可能为以下情况之一：
    * p 和 qq 在 root 的子树中，且分列 root 的 异侧（即分别在左、右子树中）；
    * p = root ，且 q 在 root 的左或右子树中；
    * q = root ，且 p 在 root 的左或右子树中；

* 终止条件：
    * 当越过叶节点，则直接返回 null ；
    * 当 root 等于 p, q，则直接返回 root ；
* 递推工作：
    * 开启递归左子节点，返回值记为 left ；
    * 开启递归右子节点，返回值记为 right ；
* 返回值： 根据 left 和 right ，可展开为四种情况；
    * 当 `left 和 right` 同时为空 ：说明 root 的左 / 右子树中都不包含 p,q ，返回 null ；
    * 当 `left 和 right` 同时不为空 ：说明 p, q 分列在 root 的 异侧 （分别在 左 / 右子树），因此 root 为最近公共祖先，返回 root ；
    * 当 `left 为空 ，right` 不为空 ：p,q 都不在 root 的左子树中，直接返回 right 。具体可分为两种情况：
        * `p,q` 其中一个在 root 的 右子树 中，此时 right 指向 p（假设为 p ）；
        * `p,q` 两节点都在 root 的 右子树 中，此时的 right 指向 最近公共祖先节点 ；
    * 当 left 不为空 ， right 为空 ：与情况 3. 同理；


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
        if not root or root == p or root == q: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right: return # 1.
        if not left: return right # 3.
        if not right: return left # 4.
        return root # 2. if left and right:
```

* 时间复杂度：$O(N)$
* 框架复杂度：$O(N)$
