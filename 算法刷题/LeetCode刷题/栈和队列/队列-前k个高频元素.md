
# 前k个高频元素

### 给定一个非空的整数数组，返回其中出现频率前k高的元素。

* 例如：
    * 输入：`nums = [1,1,1,2,2,3], k = 2`
    * 输出：`[1,2]`

* 具体操作为：
    * 借助哈希表统计元素的频率。
    * 维护一个元素数目为k的最小堆。
    * 每次都将新的元素与堆顶元素进行比较。
    * 如果新的元素的频率比堆顶的元素大，则弹出堆顶的元素，将新的元素添加进堆中。
    * 最终，堆中的k个元素即为前k个高频元素。


```python
import heapq

def solve(nums, k):
    hashmap = {}
    # 统计元素频率
    for i in nums:
        hashmap[i] = hashmap.get(i, 0) + 1
    
    # 对频率排序，定义一个小顶堆，大小为k
    pri_que = []
    for key, freq in hashmap.items():
        heapq.heappush(pri_que, (freq, key))
        if len(pri_que) > k:   # 如果堆的大小大于k，则队列弹出，保证堆堆大小一直为k
            heapq.heappop(pri_que)
    
    # 找出前k个高频元素，因为小顶堆先弹出堆最小的，倒序输出到数组
    result = [0] * k
    for i in range(k - 1, -1, -1):
        result[i] = heapq.heappop(pri_que)[1]
    return result
```


```python
nums = [1,1,1,2,2,3]
solve(nums, 2)
```




    [1, 2]


