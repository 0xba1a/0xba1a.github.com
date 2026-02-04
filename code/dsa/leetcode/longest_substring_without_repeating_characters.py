def clean_substring_hash(l_ptr, r_ptr, substring_hash, s):
    move_lptr_to = substring_hash[s[r_ptr]] + 1
    for c in s[l_ptr: move_lptr_to]:
        del substring_hash[c]
    return move_lptr_to


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        substring_hash = {}

        if not s:
            return 0

        l_ptr = 0
        r_ptr = 1
        substring_hash[s[l_ptr]] = 0
        max_len = 1
        substr = s[l_ptr]

        while r_ptr < len(s):
            if s[r_ptr] in substring_hash:
                if (r_ptr - l_ptr) > max_len:
                    max_len = r_ptr - l_ptr
                    substr = s[l_ptr:r_ptr]
                l_ptr = clean_substring_hash(l_ptr, r_ptr, substring_hash, s)
            substring_hash[s[r_ptr]] = r_ptr
            r_ptr += 1

        if (r_ptr - l_ptr) > max_len:
            max_len = r_ptr - l_ptr
            substr = s[l_ptr:r_ptr]

        return max_len

        