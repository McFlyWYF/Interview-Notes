
# 40. 最小的 k 个数

* 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入`4、5、1、6、2、7、3、8`这8个数字，则最小的4个数字是`1、2、3、4`。

* 例如：
    * 输入：`arr = [3,2,1], k = 2`
    * 输出：`[1,2] 或者 [2,1]`

#### 解题思路

##### 方法一：采用任意排序算法排序，返回前k个数


```python
def min_k(arr, k):
    def quick_sort(arr, l, r):
        if l >= r:
            return
        # 哨兵划分操作，以arr[l]作为基准
        i, j = l, r
        while i < j:
            while i < j and arr[j] >= arr[l]:
                j -= 1
            while i < j and arr[i] <= arr[l]:
                i += 1
            arr[i], arr[j] = arr[j], arr[i]
            
        arr[l], arr[i] = arr[i], arr[l]
        # 递归左右子数组执行哨兵划分
        quick_sort(arr, l, i - 1)
        quick_sort(arr, i +  1, r)
    quick_sort(arr, 0, len(arr) - 1)
    return arr[:k]
```


```python
arr = [2,4,1,0,3,5]
min_k(arr, 2)
```




    [0, 1]



* 时间复杂度：$O(NlogN)$
* 空间复杂度：$O(N)$

##### 方法二： 基于快速排序的数组划分

* 只需要将数组划分为最小的k个数和其他数字两部分即可。如果某次哨兵划分后是基准数正好是第k+1小的数字，那此时基准数左边的所有数字就是所求的最小的k个数。考虑在每次哨兵划分后，判断基准数在数组中的索引是否等于k，若true则直接返回此时数组的前k个数字即可。


```python
def min_k(arr, k):
    if k >= len(arr):
        return arr

    def quick_sort(l, r):
        # 哨兵划分操作，以arr[l]作为基准
        i, j = l, r
        while i < j:
            while i < j and arr[j] >= arr[l]:
                j -= 1
            while i < j and arr[i] <= arr[l]:
                i += 1
            arr[i], arr[j] = arr[j], arr[i]
        arr[l], arr[i] = arr[i], arr[l]
        if k < i:
            return quick_sort(l, i - 1)
        if k > i:
            return quick_sort(i +  1, r)
        return arr[:k]        
    return quick_sort(0, len(arr) - 1)
```


```python
arr = [2,4,1,0,3,5]
min_k(arr, 4)
```




    [1, 0, 2, 3]



* 时间复杂度：$O(N)$
* 空间复杂度：$O(logN)$
