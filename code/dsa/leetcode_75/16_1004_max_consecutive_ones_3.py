class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = right = zeros = max = 0

        while right < len(nums):
            if not nums[right]:
                zeros += 1
                if zeros > k:
                    cur_len = right - left
                    if cur_len > max:
                        max = cur_len

                    while nums[left]:
                        left += 1
                    left += 1
                    zeros -= 1
            right += 1
            
        cur_len = right - left
        if cur_len > max:
            max = cur_len
        return max
