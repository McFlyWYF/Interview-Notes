# 转换为str，进行切片
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x < 0:
            x = -x
            str_x = str(x)
            str_x = str_x[::-1]
            x = (-1) * int(str_x)

        else:
            str_x = str(x)
            str_x = str_x[::-1]
            x = int(str_x)

        if x <= -2 ** 31 or x > 2 ** 31 - 1:
            x = 0

        return x

# 整数操作
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        flag = 0
        rev = 0
        if x > -10 and x < 10:
            return x

        if x < 0:
            x = -x
            flag = -1
        else:
            flag = 1

        while x != 0:
            temp = x % 10
            rev = rev * 10 + temp
            x /= 10

        rev = rev * flag
        if rev <= -2 ** 31 or rev > 2 ** 31 - 1:
            rev = 0

        return rev

