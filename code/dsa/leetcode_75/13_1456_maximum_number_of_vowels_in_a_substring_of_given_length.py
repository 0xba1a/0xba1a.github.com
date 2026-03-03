class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}

        left = 0
        right = k-1

        n_vowels = 0
        for c in s[:k]:
            if c in vowels:
                n_vowels += 1

        if n_vowels == k:
            return k

        max_n_vowels = n_vowels

        while right < len(s)-1:
            if s[left] in vowels:
                n_vowels -= 1
            if s[right+1] in vowels:
                n_vowels += 1

            if n_vowels > max_n_vowels:
                if n_vowels == k:
                    return k
                max_n_vowels = n_vowels

            left += 1
            right += 1

        return max_n_vowels

