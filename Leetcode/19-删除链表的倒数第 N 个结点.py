# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# 计算链表长度，转换为删除第i个节点的问题。
# 时间复杂度O(1)，空间复杂度O(1)
class Solution1(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        list_len = 0
        p = head
        q = head
        i = 1
        while p:
            list_len += 1
            p = p.next

        index = list_len - n + 1

        # 要删除的是头节点
        if n == list_len:
            return head.next

        # 删除的是其余节点
        while q.next and i < index:
            z = q
            q = q.next
            i += 1

        z.next = q.next
        return head

# 栈
'''
先入栈，再通过出栈计数器删除该节点
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution2(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        list_len = 0
        y = head
        while y:
            list_len += 1
            y = y.next

        if n == list_len:
            return head.next

        stack = list()
        p = head
        while p:
            stack.append(p)
            p = p.next

        i = 0

        while i < n:
            i += 1
            z = stack.pop()
            q = stack[-1]

        q.next = z.next

        return head

'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        # 在第一个节点前添加一个节点，防止删除头节点时出错
        dummy = ListNode(0, head)
        stack = list()
        p = dummy
        while p:
            stack.append(p)
            p = p.next

        for i in range(n):
            stack.pop()
        
        q = stack[-1]
        q.next = q.next.next
        
        return dummy.next
'''

# 快慢指针
'''
由于我们需要找到倒数第n个节点，因此我们可以使用两个指针first和second 同时对链表进行遍历，并且first比second超前 n 个节点。
当first遍历到链表的末尾时，second 就恰好处于倒数第 n 个节点。

具体地，初始时 first 和 second 均指向头节点。我们首先使用 first 对链表进行遍历，遍历的次数为 n。
此时，first 和 second 之间间隔了 n−1 个节点，即 first 比 \second 超前了 n 个节点。

在这之后，我们同时使用 first 和 second 对链表进行遍历。当 first 遍历到链表的末尾（即 first 为空指针）时，second 恰好指向倒数第 n 个节点。
根据方法一和方法二，如果我们能够得到的是倒数第 nn 个节点的前驱节点而不是倒数第 nn 个节点的话，删除操作会更加方便。因此我们可以考虑在初始时将 
second 指向哑节点，其余的操作步骤不变。这样一来，当 first 遍历到链表的末尾时，second 的下一个节点就是我们需要删除的节点。
'''
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution3(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        dummy = ListNode(0, head)
        first = head
        second = dummy

        for i in range(n):
            first = first.next

        while first:
            second = second.next
            first = first.next

        second.next = second.next.next

        return dummy.next