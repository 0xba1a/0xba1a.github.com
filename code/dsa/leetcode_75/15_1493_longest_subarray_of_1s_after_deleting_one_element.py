class Seq:
    len: int
    zero_count: int

    def __init__(self, len, zero_count):
        self.len = len
        self.zero_count = zero_count


class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        left = 0
        right = 0

        max_seq = Seq(0, 0)
        cur_seq = Seq(0, 0)

        while right < len(nums):
            if nums[right]:
                cur_seq.len += 1
                if cur_seq.len > max_seq.len:
                    max_seq.len = cur_seq.len
                    max_seq.zero_count = cur_seq.zero_count

            else:
                if not cur_seq.zero_count:
                    cur_seq.len += 1
                    cur_seq.zero_count = 1
                    if cur_seq.len > max_seq.len:
                        max_seq.len = cur_seq.len
                        max_seq.zero_count = cur_seq.zero_count

                else:
                    while nums[left]:
                        left += 1
                    left += 1 # shift next to the zero
                    cur_seq.len = right - left + 1 #both inclusive

            right += 1

        return max_seq.len - 1
    