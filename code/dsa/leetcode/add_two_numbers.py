# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

def construct_number(lst):
    number = 0
    decimal_pos = 1
    while lst.next != None:
        number += decimal_pos * lst.val
        decimal_pos *= 10
        lst = lst.next

    number += decimal_pos * lst.val
    print(number)
    return number


def construct_linked_list(lst):
    prev_node = None
    for item in lst:
        node = ListNode(item, prev_node)
        prev_node = node
    if not prev_node:
        return ListNode(0, None)
    return prev_node


def construct_list(num):
    lst = []
    print("Num", num)
    while num > 0:
        item = num % 10
        print("item", item)
        num = num // 10
        print("num:", num)
        lst.insert(0, item)

    print(lst)
    return construct_linked_list(lst)



class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        l = construct_number(l1)
        r = construct_number(l2)
        return construct_list(l + r)
    

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution1:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        carry = 0
        head = None
        prev_node = None

        while l1 or l2:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            sum = val1 + val2 + carry
            carry = sum // 10
            val = sum % 10
            cur_node = ListNode(val, None)

            if not head:
                head = cur_node

            if prev_node:
                prev_node.next = cur_node

            prev_node = cur_node

            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        if carry:
            cur_node = ListNode(carry, None)
            prev_node.next = cur_node

        if not head:
            head = ListNode(0, None)

        return head