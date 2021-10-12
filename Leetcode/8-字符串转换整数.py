# 判断语句
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        flag = 1
        temp = 0
        s = s.lstrip()  # 丢弃无用的前导空格

        for i, v in enumerate(s):
            if v >= '0' and v <= '9':   # 前面是'-'或'+'或数字
                temp = temp * 10 + int(v)
            elif v == '-' and i == 0:   # 第一个是'-'
                flag = -1
            elif v == '+' and i == 0:   # 第一个是'+'
                flag = 1
            else:
                break
        x = flag * temp
        # 判断是否越界
        if x < -2**31:
            return -2**31
        elif x > 2**31-1:
            return 2**31-1
        return x


# 有限状态机
INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31

class Automaton:
    def __init__(self):
        self.state = 'start'
        self.sign = 1
        self.ans = 0
        self.table = {
            'start': ['start', 'signed', 'in_number', 'end'],
            'signed': ['end', 'end', 'in_number', 'end'],
            'in_number': ['end', 'end', 'in_number', 'end'],
            'end': ['end', 'end', 'end', 'end'],
        }

    def get_col(self, c):
        if c.isspace():
            return 0
        if c == '+' or c == '-':
            return 1
        if c.isdigit():
            return 2
        return 3

    def get(self, c):
        self.state = self.table[self.state][self.get_col(c)]
        if self.state == 'in_number':
            self.ans = self.ans * 10 + int(c)
            self.ans = min(self.ans, INT_MAX) if self.sign == 1 else min(self.ans, -INT_MIN)
        elif self.state == 'signed':
            self.sign = 1 if c == '+' else -1


class Solution1(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        automaton = Automaton()
        for c in s:
            automaton.get(c)
        return automaton.sign * automaton.ans


str = '   -42'
s = Solution1()
print(s.myAtoi(str))

