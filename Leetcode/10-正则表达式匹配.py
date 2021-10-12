# 动态规划求解
'''
使用状态转移方程；
p[j] == s[i]：dp[i][j] = dp[i-1][j-1]，当前字符可以匹配上；
p[j] == '.'：dp[i][j] = dp[i-1][j-1]，当前可以匹配任意一个字符；
p[j] == '*'：
    p[j-1] != s[i]：dp[i][j] = dp[i][j-2]：s的当前字符和*的前一个字符不匹配，就相当于匹配0个，扔掉p的这2个字符；
    p[   ：也就是s的当前字符可以匹配*前的字符或者*前的字符是'.'，匹配成功；
        dp[i][j] = dp[i-1][j] or    多个字符匹配的情况
        dp[i][j] = dp[i][j-1] or    单个字符匹配的情况
        dp[i][j] = dp[i][j-2]       没有字符匹配的情况
'''
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        n, m = len(s), len(p)
        f = [[False] * (m + 1) for _ in range(n + 1)]
        f[0][0] = True  # s和p都为空时，匹配

        # 当s为空，p不空时，要看p是否为a*b*这种结构
        for j in range(1, m + 1):
            if p[j - 1] == '*':
                f[0][j] = f[0][j - 2]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if p[j - 1] == '*':
                    if p[j - 2] in (s[i - 1], '.'):
                        f[i][j] = f[i - 1][j] or f[i][j - 2]
                    else:
                        f[i][j] = f[i][j - 2]

                # 单个字符匹配情况
                else:
                    if p[j - 1] in (s[i - 1], '.'):
                        f[i][j] = f[i - 1][j - 1]
        return f[-1][-1]

s = "aab"
p = "c*a*b"
s1 = Solution()
ss = s1.isMatch(s, p)
print(ss)