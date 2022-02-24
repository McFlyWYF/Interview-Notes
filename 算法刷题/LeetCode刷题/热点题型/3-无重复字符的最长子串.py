# 滑动窗口方法
'''
其实就是一个队列,比如例题中的 abcabcbb，进入这个队列（窗口）为 abc 满足题目要求，
当再进入 a，队列变成了 abca，这时候不满足要求。所以，我们要移动这个队列！
如何移动？
我们只要把队列的左边的元素移出就行了，直到满足题目要求！
一直维持这样的队列，找出队列出现最长的长度时候，求出解！
'''

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == "":
            return 0
        left = 0
        lookup = set()  # 相当于队列，存储元素
        length = len(s)
        max_len = 0
        cur_len = 0
        for i in range(length):
            cur_len += 1
            while s[i] in lookup:   # 判断是否存在相同字符
                lookup.remove(s[left])  # 移除最左边的元素
                left += 1
                cur_len -= 1

            if cur_len > max_len:
                max_len = cur_len
            lookup.add(s[i])
        return max_len

# 暴力法

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0

        max_len = 0

        def search(string, max_len):
            hashmap = dict()
            cnt = 0
            for i in string:
                if not i in hashmap:
                    hashmap[i] = i
                    cnt += 1
                else:
                    break
            if cnt > max_len:
                max_len = cnt
            return max_len

        for i in range(len(s)):
            max_len = search(s[i:], max_len)
        return max_len
