# 用字典实现，小的在左边，减去，小的在右边，加上
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        num_sum = 0
        map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        for i in range(len(s) - 1):
            if map[s[i]] >= map[s[i + 1]]:
                num_sum += map[s[i]]
            else:
                num_sum -= map[s[i]]

        num_sum += map[s[-1]]
        return num_sum

# 切割字符串，寻找是否存在子串
class Solution1(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000, 'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
        num_sum = 0
        i = 0
        while i < len(s):
            sub_str = s[i:i + 2]
            if sub_str in map:
                num_sum += map.get(sub_str)
                i += 2
            else:
                num_sum += map.get(s[i])
                i += 1
        return num_sum

str = "IV"
s = Solution1()
print(s.romanToInt(str))