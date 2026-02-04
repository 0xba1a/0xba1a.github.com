class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return i, j
        return -1, -1
    
class Solution1:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen_nums = {}

        for index, i in enumerate(nums):
            complement = target - i
            if complement in seen_nums:
                return index, seen_nums[complement]

            seen_nums[i] = index