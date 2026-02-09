class Solution:
    def myAtoi(self, s: str) -> int:
        if len(s) == 0:
            return 0

        INT_MAX = 2**31-1
        INT_MIN = -2**31

        s = s.strip()

        sign = 1
        can_see_sign = True
        num = 0
        SIGNS = ['-', '+']

        for c in s:
            if c in SIGNS:
                if not can_see_sign: # Second sign symbol
                    break
                elif c == '-':
                    sign = -1
                    can_see_sign = False
                else:
                    can_see_sign = False
                continue

            try:
                i = int(c)
                if sign == -1 and num*sign < (INT_MIN+i) // 10:
                    return INT_MIN
                elif sign == 1 and num > (INT_MAX-i) // 10:
                    return INT_MAX
                num = num * 10 + i
                can_see_sign = False
            except:
                break

        return num*sign


            
sol = Solution()
print(sol.myAtoi("-2147483649"))