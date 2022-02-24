# 两两进行比较找公共子串，也称为横向扫描
'''
当字符串数组长度为 0 时则公共前缀为空，直接返回
令最长公共前缀 ans 的值为第一个字符串，进行初始化
遍历后面的字符串，依次将其与 ans 进行比较，两两找出公共前缀，最终结果即为最长公共前缀
如果查找过程中出现了 ans 为空的情况，则公共前缀不存在直接返回
时间复杂度：O(s)O(s)，s 为所有字符串的长度之和
'''
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ""
        str0 = strs[0]
        for i in range(1, len(strs)):
            j = 0
            while j < len(str0) and j < len(strs[i]):
                if str0[j] != strs[i][j]:
                    break
                j += 1
            str0 = str0[0:j]
            if str0 == "":
                return str0
        return str0

# 纵向扫描，比较每个字符串对应位置的字符
class Solution1(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if strs == "":
            return ""

        str0_len, num = len(strs[0]), len(strs)

        for i in range(str0_len):
            c = strs[0][i]

            if any(i == len(strs[j]) or strs[j][i] != c for j in range(1, num)):
                return strs[0][0:i]
        return strs[0]

# 分治法
class Solution2(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if strs == "":
            return ""

        def lcp(start, end):
            if start == end:
                return strs[start]

            middle = start + end // 2
            left, right = lcp(start, middle), lcp(middle + 1, end)
            min_len = min(len(left), len(right))

            for i in range(min_len):
                if left[i] != right[i]:
                    return left[:i]

            return left[:min_len]

        return lcp(0, len(strs) - 1)

# 二分法
class Solution3(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """

        def isCommonPrefix(length):
            str0, count = strs[0][:length], len(strs)
            return all(strs[i][:length] == str0 for i in range(1, count))

        if not strs:
            return ""

        minLength = min(len(s) for s in strs)
        low, high = 0, minLength
        while low < high:
            mid = (high - low + 1) // 2 + low
            if isCommonPrefix(mid):
                low = mid
            else:
                high = mid - 1

        return strs[0][:low]


strs = ["flower","flow", "flight"]
s = Solution3()
print(s.longestCommonPrefix(strs))
