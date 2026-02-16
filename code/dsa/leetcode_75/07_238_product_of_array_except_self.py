class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        left_product = [1] * length
        right_product = [1] * length

        # i in left_product should represent left_product until i-1 inclusive
        # i in right_product should represent right_product until i+1 inclusive

        for i in range(1, length):
            left_product[i] = left_product[i-1] * nums[i-1]
            right_index = length - 1 - i
            right_product[right_index] = right_product[right_index+1] * nums[right_index+1]
        right_product[0] = right_product[1] * nums[1]

        # print(left_product)
        # print(right_product)

        res = []
        for i in range(length):
            res.append(right_product[i] * left_product[i])

        return res