
# 四数相加2

### 给定两个包含整数的数组列表A, B, C, D, 计算有多少个元组(i, j, k, l)，使得A[i]+B[j]+C[k]+D[l]=0。

* 例如：
    * 输入：A=[1,2], B=[-2,-1], C=[-1,2], D=[0,2]
    * 输出：2

#### 解题步骤
* 首先定义一个map，key放a和b两数之和，value放a和b两数之和出现的次数。
* 遍历大A和大B数组，统计两个数组元素之和，和出现的次数，放到map中。
* 定义变量count，用来统计a+b+c+d=0出现的次数。
* 在遍历大C和大D数组，找到如果0-(c+d)在map中出现过的话，就用count把map中key对应的value也就是出现的次数统计出来。
* 最后返回统计值count就可以了。


```python
def solve(A,B,C,D):
    sum_map = dict()
    for a in A:
        for b in B:
            if a + b in sum_map:
                sum_map[a+b] += 1
            else:
                sum_map[a+b] = 1
    count = 0
    for c in C:
        for d in D:
            if (-1) * (c+d) in sum_map:
                count += sum_map[0-c-d]
    return count
```


```python
A = [1, 2]
B = [-2, -1]
C = [-1, 2]
D = [0, 2]
solve(A, B, C, D)
```




    2


