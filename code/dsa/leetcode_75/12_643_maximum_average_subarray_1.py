class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        if not k:
            return 0

        left = 0
        right = k-1

        sum = 0
        for n in nums[:k]:
            sum += n

        max_sum = sum
        while right < len(nums)-1:
            sum -= nums[left]
            sum += nums[right+1]

            if sum > max_sum:
                max_sum = sum

            left += 1
            right += 1

        return max_sum / k