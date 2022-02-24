# 栈
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = list()
        left = ['(', '[', '{']

        for i in range(len(s)):
            if s[i] in left:
                stack.append(s[i])
            elif s[i] == ')' and stack and stack[-1] == '(':
                stack.pop()
            elif s[i] == ']' and stack and stack[-1] == '[':
                stack.pop()
            elif s[i] == '}' and stack and stack[-1] == '{':
                stack.pop()
            else:
                return False

        if stack:
            return False
        else:
            return True

# 哈希表（存储括号对）+栈
class Solution1(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        hashmap = {
            ')': '(',
            ']': '[',
            '}': '{'
        }

        stack = list()
        for ch in s:
            if stack and ch in hashmap:
                if stack[-1] == hashmap[ch]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(ch)

        return not stack

str =  "()"
s = Solution1()
print(s.isValid(str))