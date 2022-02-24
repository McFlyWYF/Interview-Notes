
# 22. 链表中倒数第k个节点

* 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

* 例如：
    * 给定一个链表: `1->2->3->4->5, 和 k = 2`.
    * 返回链表 `4->5`.

#### 解题思路

* 创建两个指针，快指针fast和慢指针slow。
* 让快指针指向距离头结点k个位置的节点。
* 同时移动快慢指针，直到快指针为空为止，此时慢指针指向的就是倒数第k个节点。


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getKthFromEnd(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """

        fast = head
        slow = head
        while k > 0 and fast:
            fast = fast.next
            k -= 1
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
```
