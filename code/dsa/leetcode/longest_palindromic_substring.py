def get_max_pal_str(s, i, j):
    max_pal_str = ""

    while i >= 0 and j < len(s) and s[i] == s[j]:
        max_pal_str = s[i:j+1]
        i -= 1
        j += 1

    return max_pal_str

class Solution:
    def longestPalindrome(self, s: str) -> str:
        max_pal_str = ""
        for index in range(len(s)):
            i = j = index
            odd_pal_str = get_max_pal_str(s, i, j)
            if len(odd_pal_str) > len(max_pal_str):
                max_pal_str = odd_pal_str

            i = index
            j = index+1
            even_pal_str = get_max_pal_str(s, i, j)
            if len(even_pal_str) > len(max_pal_str):
                max_pal_str = even_pal_str

        return max_pal_str