
# 两个数组的交集

### 给定两个数组，编写一个函数来计算它们的交集。

* 例如：
    * 输入：nums1 = [1,2,2,1], nums2=[2,2]
    * 输出：[2]

* C++中的unordered_set底层实现是哈希表，读写效率是最高的，并不需要对数据进行排序，而且还不让数据重复。

![FCF1E24B8C7B13B4557B2B1314B5E7C9.png](attachment:3135b17f-4a99-41d4-95cc-b4066c22c3ae.png)


```python
def intersection(nums1, nums2):
    result_set = set()
    set1 = set(nums1)  # 保存为set
    
    for n2 in nums2:
        if n2 in set1:   # 如果在nums1中
            result_set.add(n2)
    return list(result_set)
```


```python
nums1 = [1,2,2,1]
nums2 = [2,2]
intersection(nums1, nums2)
```




    [2]


