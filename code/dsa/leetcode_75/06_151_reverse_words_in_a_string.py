class Solution:
    def reverseWords(self, s: str) -> str:
        words = list(s.strip().split())

        pos = 0

        for word in words[::-1]:
            words.insert(pos, word)
            pos += 1

        return " ".join(words[:pos])
