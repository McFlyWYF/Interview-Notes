
# 32 - II. 从上到下打印二叉树 II

* 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

* 例如:给定二叉树: `[3,9,20,null,null,15,7]`
* 输出：`[
  [3],
  [9,20],
  [15,7]
]`

### 递归法


```python
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def transvel(cur, depth):
            if len(result) == depth:
                result.append([])
            result[depth].append(cur.val)
            if cur.left:
                transvel(cur.left,depth+1)
            if cur.right:
                transvel(cur.right,depth+1)
        
        if root == None:
            return []
        result = []
        transvel(root, 0)
        return result
```

### 队列法


```python
class Solution(object):
    def levelOrder(self, root):
        q = deque([root])
        result = []
        if root == None:
            return result
        while q:
            size = len(q)
            path = []
            for _ in range(size):
                cur = q.popleft()
                path.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            result.append(path)
        return result
```
