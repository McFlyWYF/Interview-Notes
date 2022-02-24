
# 32 - III. 从上到下打印二叉树 III

* 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

* 给定二叉树: `[3,9,20,null,null,15,7]`
* `[
  [3],
  [20,9],
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

        for i in range(1, len(result), 2):
            result[i] = result[i][::-1]

        return result
```

### 队列法


```python
class Solution(object):
    def levelOrder(self, root):
        q = deque()
        result = []
        if root == None:
            return result
        q = deque([root])
        j = 0
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
            if j % 2 == 1:
                result.append(path[::-1])
            else:
                result.append(path)
            j += 1
        return result
            
```
