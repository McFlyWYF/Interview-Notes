# II-剪绳子II

* 给你一根长度为n的绳子，请把绳子剪成整数长度的m段，每段绳子的长度记为`k[0],k[1]...k[m-1]`。请问`k[0]*k[1]*...*k[m-1]`可能的最大乘积是多少？


```python
def cuttingRope(n):
    """
    :type n: int
    :rtype: int
    """
    if n <= 3:
        return n - 1
    b = n % 3
    if b == 0:
        return int(pow(3, n // 3) % 1000000007)
    if b == 1:
        return int((pow(3, n // 3 - 1) * 4) % 1000000007)
    if b == 2:
        return int((pow(3, n // 3) * 2) % 1000000007)
```


```python
cuttingRope(120)
```




    953271190



* 时间复杂度：$O(1)$
* 空间复杂度：$O(1)$
