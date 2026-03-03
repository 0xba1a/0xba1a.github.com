class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max = 0

        while left < right:
            w = right - left
            if height[left] < height[right]:
                area = w * height[left]
                if area > max:
                    max = area
                left += 1
            else:
                area = w * height[right]
                if area > max:
                    max = area
                right -= 1

        return max