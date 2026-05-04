class Solution:
    def removeStars(self, s: str) -> str:
        stack: list = list(s)
        top = 0

        for c in s:
            if c == '*':
                top -= 1

            else:
                stack[top] = c
                top += 1

        return "".join(stack[:top])