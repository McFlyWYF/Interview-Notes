
# 55 - II. 平衡二叉树

* 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

* 例如：
    * 给定二叉树 `[3,9,20,null,null,15,7]`
    * 返回 `true`

#### 解题思路

##### 方法一：递归法

* 返回值
    * 当节点root左右子树的深度差<=1，返回当前子树的深度，即左右子树深度最大值+1
    * 当节点root左右子树的深度差>2，返回-1，代表不是平衡树
* 终止条件
    * 当root为空，说明越过叶节点，返回高度0
    * 当左右子树深度为-1，代表此树的左右子树不是平衡树，因此剪枝，返回-1


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def getdepth(root):
            if not root:
                return 0
            left = getdepth(root.left)
            if left == -1:
                return -1
            right = getdepth(root.right)
            if right == -1:
                return -1
            return max(left, right) + 1 if abs(left - right) <= 1 else -1
        return getdepth(root) != -1
```

##### 方法二：迭代法


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        def getdepth(cur):
            # 后序遍历每一个节点的高度，将当前传入的节点作为根节点
            st = []
            if cur:
                st.append(cur)
            depth = 0
            result = 0
            while st:
                node = st.pop()
                if node:
                    st.append(node)
                    st.append(None)
                    depth += 1
                    if node.right:
                        st.append(node.right)
                    if node.left:
                        st.append(node.left)
                else:
                    node = st.pop()
                    depth -= 1
                result = max(result, depth)
            return result

        # 前序遍历每个节点，判断左右孩子高度是否符合
        if root == None:
            return True
        stack = [root]
        while stack:
            node = stack.pop()  # 根
            if abs(getdepth(node.left) - getdepth(node.right)) > 1:
                return False
            if node.right:  # 右
                stack.append(node.right)
            if node.left:  # 左
                stack.append(node.left)
            
        return True
```
