
# 重复的子字符串

### 给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过1000。

* 例如：
    * 输入：'abab'
    * 输出：True

最长相等前后缀的长度为：`next[len -1] + 1`，如果`len % (len - (next[len - 1] + 1)) == 0`，则说明（数组长度-最长相等前后缀的长度）正好可以被数组的长度整除，说明该字符串有重复的子字符串。

#### 前后缀减一


```python
def getnext(next, s):
    next[0] = -1
    j = -1
    for i in range(1, len(s)):
        while j >= 0 and s[i] != s[j+1]:
            j = next[j]
        if s[i] == s[j+1]:
            j += 1
        next[i] = j
    return next
```


```python
def solve(s):
    if len(s) == 0:
        return False
    next = [0 for _ in range(len(s))]
    getnext(next, s)
    print(next)
    if next[-1] != -1 and len(s) % (len(s) - (next[-1] + 1)) == 0:
        return True
    return False
```


```python
s = 'abcab'
solve(s)
```

    [-1, -1, -1, 0, 1]
    




    False




```python
s = 'abab'
solve(s)
```

    [-1, -1, 0, 1]
    




    True



#### 前后缀不减一


```python
def getnext(next, s):
    next[0] = -1
    j = -1
    for i in range(1, len(s)):
        while j > 0 and s[i] != s[j]:
            j = next[j]
        if s[i] == s[j]:
            j += 1
        next[i] = j
    return next
```


```python
def solve(s):
    if len(s) == 0:
        return False
    next = [0 for _ in range(len(s))]
    getnext(next, s)
    print(next)
    if next[-1] != -1 and len(s) % (len(s) - (next[-1] + 1)) == 0:
        return True
    return False
```


```python
s = 'abab'
solve(s)
```

    [-1, 0, 1, 2]
    




    True


