class Solution:
    def reverse(self, x: int) -> int:

        rev = 0
        sign = 1
        INT_MAX = 2**31 - 1
    
        if x < 0:
            sign = -1
            x *= -1

        while x > 0:
            if rev > INT_MAX / 10:
                return 0
            rev = (rev << 3) + (rev << 1) + x % 10
            x //= 10

        if rev & 0b1<<31:
            return 0

        return rev*sign
    

class Solution1:
    def reverse(self, x:int) -> int:
        rev, sign = 0, 1
        if x < 0:
            sign = -1
            x *= -1

        while x > 0:
            rev = rev * 10 + x % 10
            x //=10

        if rev >= 0b1<<31:
            return 0
        return rev*sign