
# 32 - I. 从上到下打印二叉树

* 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

例如:给定二叉树: `[3,9,20,null,null,15,7]`

返回：`[3,9,20,15,7]`

### 递归法


```python
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
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
        res = []
        for i in result:
            for j in i:
                res.append(j)
        return res
```

### 队列法


```python
class Solution(object):
    def levelOrder(self, root):
        result = []
        if root == None:
            return result
        q = deque([root])
        while q:
            size = len(q)
            for _ in range(size):
                cur = q.popleft()
                result.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return result
```
