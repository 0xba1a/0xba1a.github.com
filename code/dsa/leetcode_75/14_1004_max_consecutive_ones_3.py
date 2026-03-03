class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = right = total_1s_in_sw = sw_len = 0
        max_len = 0

        while right < len(nums):
            sw_len = right - left + 1

            if nums[right]:
                total_1s_in_sw += 1

            if total_1s_in_sw + k >= sw_len:
                if sw_len > max_len:
                    max_len = sw_len
            else:
                if nums[left]:
                    total_1s_in_sw -= 1
                left += 1

            right += 1

        return max_len
