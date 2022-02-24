
# 实现strStr()

### 实现strStr()函数。给定一个haystack字符串和一个needle字符串，在haystack字符串中找出needle字符串中出现的第一个位置。如果不存在，则返回-1。

* 例如：
    * 输入：haystack=‘hello',needle='ll'
    * 输出：2

当needle字符串是空字符串时，我们返回0。

#### 构造next数组

1. 初始化
2.处理前后缀不相同的情况
3.处理前后缀相同的情况

##### 1.初始化

`j`指向前缀终止位置，`i`指向后缀终止位置，`j`初始化为`-1`。
```python
next[0] = j
```

##### 2.处理前后缀不相同的情况

因为j初始化为-1，`i`就从`1`开始，进行`s[i]`与`s[j+1]`的比较。

如果`s[i]`与`s[j+1]`不相同，就是遇到前后缀末尾不相同的情况，就要向前回溯。

next[j]就是记录j之前的子串的相同前后缀的长度。那么s[i]和s[j+1]不相同，就要找j+1前一个元素在next数组里的值。

```python
while j >= 0 and s[i] != s[j + 1]:
    j = next[j]
```

##### 3.处理前后缀相同的情况

如果s[i]和s[j+1]相同，那么就同时向后移动`i`和`j`，说明找到了相同的前后缀，同时还要将j赋给next[i]，因为next[i]要记录相同前后缀的长度。

```python
if s[i] == s[j+1]:
    j += 1
next[i] = j
```


```python
def getNext():
    j = -1
    next[0] = j
    for i in range(1, len(s)):
        while j >= 0 and s[i] != s[j + 1]:   # 前后缀不同
            j = next[j]
        if s[i] == s[j+1]:   # 相同的前后缀
            j += 1
        next[i] = j
```

#### 使用next数组来做匹配

定义两个下标j指向模式串t起始位置，i指向文本串s起始位置。j初始值依然为-1，i从0开始，遍历文本串。
```python
for i in range(len(s))
```

接下来就是s[i]与t[j+1]比较。如果s[i]与t[j+1]不相同，j就要从next数组里寻找下一个匹配的位置。

```python
while j >= 0 and s[i] != t[j+1]:
    j = next[j
```

如果s[i]与t[j+1]相同，那么i和j同时向后移动。
```python
if s[i] == t[j+1]:
    j += 1
```

如果j指向了模式串t的末尾，那么就说明模式串t完全匹配文本串s里的某个子串了。

```python
if j == len(t) - 1:
    return i - len(t) + 1
```


```python
def KMP():
    j = -1
    for i in range(len(s)):
        while j >= 0 and s[i] != t[j+1]:   # 不匹配
            j = next[j]
        if s[i] == t[j+1]:   # 匹配，i和j同时后移
            j += 1
        if j == (len(t) - 1):
            return i - len(t) + 1
```

#### 代码实现

##### 前后缀减一


```python
def getnext(a,neddle):
    next = ['' for i in range(a)]
    k = -1
    next[0] = k
    for i in range(1, len(neddle)):
        while k >= 0 and neddle[i] != neddle[k + 1]:   # 前后缀不同
            k = next[k]
        if neddle[i] == neddle[k+1]:   # 相同的前后缀
            k += 1
        next[i] = k
    return next
```


```python
def KMP(haystack, neddle):
    a = len(haystack)
    b = len(neddle)
    if b == 0:
        return 0
    next = getnext(b,neddle)
    j = -1
    for i in range(a):
        while j >= 0 and haystack[i] != neddle[j+1]:   # 不匹配
            j = next[j]
        if haystack[i] == neddle[j+1]:   # 匹配，i和j同时后移
            j += 1
        if j == (b - 1):
            return i - b + 1
    return -1
```


```python
haystack = 'hello'
needle = 'll'
KMP(haystack, needle)
```




    2



##### 前后缀不减一


```python
def getnext(a,neddle):
    next = ['' for i in range(a)]
    k = -1
    next[0] = k
    for i in range(1, len(neddle)):
        while k > 0 and neddle[i] != neddle[k]:   # 前后缀不同
            k = next[k]
        if neddle[i] == neddle[k]:   # 相同的前后缀
            k += 1
        next[i] = k
    return next
```


```python
def KMP(haystack, neddle):
    a = len(haystack)
    b = len(neddle)
    if b == 0:
        return 0
    next = getnext(b,neddle)
    j = -1
    for i in range(a):
        while j > 0 and haystack[i] != neddle[j]:   # 不匹配
            j = next[j]
        if haystack[i] == neddle[j]:   # 匹配，i和j同时后移
            j += 1
        if j == (b - 1):
            return i - b + 1
    return -1
```


```python
haystack = 'hello'
needle = 'll'
KMP(haystack, needle)
```




    2


