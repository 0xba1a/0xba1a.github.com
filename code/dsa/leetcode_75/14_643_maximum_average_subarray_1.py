class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        if not k:
            return 0

        sum = 0
        for n in nums[:k]:
            sum += n

        left = 1
        right = k

        max_sum = sum
        while right < len(nums):
            sum -= nums[left]
            sum += nums[right]

            if sum > max_sum:
                max_sum = sum

            left += 1
            right += 1

        return max_sum / k