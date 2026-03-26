class Solution:
    def convert_int_to_chars(self, seq):
        return list(str(seq))

    def compress(self, chars: List[str]) -> int:
        read = write = 0

        cur_char = chars[0]
        read = 1
        seq_len = 1

        while read < len(chars):
            if chars[read] != cur_char:
                chars[write] = cur_char
                write += 1
                if seq_len > 1:
                    seq_in_chars = self.convert_int_to_chars(seq_len)
                    for c in seq_in_chars:
                        chars[write] = c
                        write += 1

                cur_char = chars[read]
                seq_len = 1
            else:
                seq_len += 1
            read += 1
        
        chars[write] = cur_char
        write += 1
        if seq_len > 1:
            seq_in_chars = self.convert_int_to_chars(seq_len)
            for c in seq_in_chars:
                chars[write] = c
                write += 1

        return write
