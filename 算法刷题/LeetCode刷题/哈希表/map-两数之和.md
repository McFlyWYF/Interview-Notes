
# 两数之和

### 给定一个整数数组nums和一个目标值target，在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。

* 例如：
    * 输入：nums = [2, 7, 11, 15], target = 9
    * 输出：[0, 1]

使用暴力法是两层for循环，时间复杂度是`O(n^2)`。使用哈希法最为合适。

* 数组和set做哈希法的局限：
    * 数组的大小是受限制的，而且如果元素很少，而哈希值太大会造成内存空间的浪费。
    * set是一个集合，里面放的元素只能是一个key，而两数之和，不仅要判断y是否存在，而且还要记录y的下标位置。所以set不能用。


```python
def solve(nums, target):
    index_map = {}
    for index, num in enumerate(nums):
        index_map[num] = index
    for i, num in enumerate(nums):
        j = index_map.get(target - num)
        if j is not None and i != j:
            return [i, j]
```


```python
nums = [2, 7, 11, 15]
target = 9
solve(nums, target)
```




    [0, 1]


