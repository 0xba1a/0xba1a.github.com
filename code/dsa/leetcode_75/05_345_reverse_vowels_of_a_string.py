class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        i = 0
        j = len(s)-1
        res = list(s)

        while True:
            while i < len(s) and s[i] not in vowels: i += 1
            while j >= 0 and s[j] not in vowels: j -= 1

            if i >= j:
                break

            res[i], res[j] = s[j], s[i]
            i += 1
            j -= 1

        return ''.join(res)