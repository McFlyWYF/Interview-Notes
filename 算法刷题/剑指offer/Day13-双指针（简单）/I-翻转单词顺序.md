
# 58 - I. 翻转单词顺序

* 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

* 例如：
    * 输入: `"the sky is blue"`
    * 输出: `"blue is sky the"`

#### 分割+倒序


```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = s.strip()  # 去除首尾空格
        s = s.split()  # 根据空格切分字符串
        res = []
        for i in range(len(s) - 1, -1, -1):
            res.append(s[i])  # 倒序保存到res中
        return " ".join(res)
```

* 时间复杂度：$O(n)$
* 空间复杂度：$O(n)$

#### 双指针


```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = []
        s = s.strip()  # 去除首尾空格
        i = j = len(s) - 1
        while i >= 0:
            while i >= 0 and s[i] != " ":  # 找到该单词的第一个位置
                i -= 1
            res.append(s[i + 1:j + 1])  # 将该单词保存到res中
            while s[i] == " ":  # 跳过空格，遍历下一个单词
                i -= 1
            j = i  # 该单词的结束位置
        return " ".join(res)
```

* 时间复杂度：$O(n)$
* 空间复杂度：$O(n)$
