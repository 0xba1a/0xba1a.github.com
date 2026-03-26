class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        left = 0
        right = len(nums)

        while left < right:
            if nums[left]:
                left += 1
                continue

            nums.insert(right, nums[left])
            del nums[left] # del should happen only after the insert
            right -= 1
        