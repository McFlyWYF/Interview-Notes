class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        # 朴素匹配算法
        if not needle:
            return 0

        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i : i + len(needle)] == needle:
                return i

        return -1

        # API
        return haystack.find(needle)

        # KMP

h = "hello"
n = 'lldsadsad'

s = Solution()
print(s.strStr(h, n))