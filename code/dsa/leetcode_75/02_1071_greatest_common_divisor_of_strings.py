class Solution:
    def is_divide(self, str1, str2, prefix):
        # prefix cannot be lengthier than str2
        str2_len = len(str2)
        prefix_len = len(prefix)
        if not prefix_len:
            return True

        if str2_len % prefix_len or len(str1) % prefix_len:
            return False

        for i in range(len(str1)):
            prefix_char = prefix[i % prefix_len]
            if str1[i] != prefix_char or str2[i % str2_len] != prefix_char:
                return False
        return True

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if len(str1) < len(str2):
            str1, str2 = str2, str1

        # str2 is always smaller or equal length
        prefixes = ['']

        for i in range(len(str2)):
            if str1[i] == str2[i]:
                prefixes.append(prefixes[-1] + str1[i])

        for prefix in prefixes[::-1]:
            if self.is_divide(str1, str2, prefix):
                return prefix
