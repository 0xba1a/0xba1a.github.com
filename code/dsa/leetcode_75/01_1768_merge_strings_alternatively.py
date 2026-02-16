class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        final_str = ''
        i = 0
        while i < len(word1) and i < len(word2):
            final_str += word1[i] + word2[i]
            i += 1

        if i < len(word1):
            final_str += word1[i:]
        elif i < len(word2):
            final_str += word2[i:]

        return final_str