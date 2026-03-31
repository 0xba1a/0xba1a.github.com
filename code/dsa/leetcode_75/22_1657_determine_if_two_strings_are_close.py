from collections import Counter

class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        w1_counter = Counter(word1)
        w2_counter = Counter(word2)

        if set(w1_counter.keys()) != set(w2_counter.keys()):
            return False

        w1_vals = list(w1_counter.values())
        w1_vals.sort()

        w2_vals = list(w2_counter.values())
        w2_vals.sort()

        if w1_vals != w2_vals:
            return False

        return True
