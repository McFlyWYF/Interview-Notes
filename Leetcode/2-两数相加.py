# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

'''
解题思路：
1. 对应元素相加，先将进位设置为0
2. 判断是否有进位
3. 元素赋值到新链表中
'''

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        carry = 0
        l3 = re = ListNode(0)   # 第一个节点赋值为0
        while l1 or l2 or carry:
            sum = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry     # 对应位置的元素相加，再加上进位
            (carry, sum) = (0, sum) if sum < 10 else (1, sum - 10)      # 如果大于10，进位为1，否则为0

            l3.next = ListNode(sum)     # 将该值赋值到链表中
            l3 = l3.next    # 指向下一个节点
            print(l3)

            l1 = l1.next if l1 else None    # 指向下一个节点
            l2 = l2.next if l2 else None    # 指向下一个节点

        return re.next

