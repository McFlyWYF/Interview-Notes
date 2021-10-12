# 暴力法（超时）
'''
找到所有的字串，判断是否为回文串
'''
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        max = 0
        str = ""
        for i in range(len(s) + 1):
            for j in range(1, len(s) + 1):
                s1 = s[i:j]
                s2 = s1[::-1]

                if(s1 == s2):
                    length = len(s1)

                if(max < length):
                    max = length
                    str = s1

        return str

# 中心扩散法
'''
从中心向两头扩散，从第一个字符开始，如果相等，left - 1，right + 1，如果不相等，移向下一个字符
'''
class Solution1:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ""

        for i in range(len(s)):
            s1 = self.find(s, i, i)     # 以当前字符为中心的最长回文字串，中心为奇数
            s2 = self.find(s, i, i + 1)     # 以当前字符和下一个字符为中心的最长回文字串，中心为偶数
            print(s1)

            if max(len(s1), len(s2)) > len(res):
                res = s2 if len(s1) < len(s2) else s1

        return res

    def find(self, s, left, right):
        # 找到当前中心的最大长度字串
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1: right]

# 动态规划
'''
从两头开始判断，如果不等，则该字串不是回文串，如果相等，则去除头尾之后，判断剩下的字串是否是回文串
'''
