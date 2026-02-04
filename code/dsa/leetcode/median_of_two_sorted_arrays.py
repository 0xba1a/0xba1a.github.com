class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # Always keeps the smaller array at nums1
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        if len(nums1) == 0 and len(nums2) == 0:
            return 0

        x = [float("-infinity")] + nums1 + [float("infinity")]
        y = [float("-infinity")] + nums2 + [float("infinity")]

        total_len = len(x) + len(y)
        l_part_len = total_len // 2

        l = 0
        r = len(x) - 1

        while True:
            m = (l + r) // 2
            x_elements = m + 1 # m is the index
            y_elements = l_part_len - x_elements

            x_left_end, x_right_start = x[x_elements-1], x[x_elements]
            y_left_end, y_right_start = y[y_elements-1], y[y_elements]

            if x_left_end <= y_right_start and y_left_end <= x_right_start:
                if total_len % 2 == 0:
                    return (max(x_left_end, y_left_end) + min(x_right_start, y_right_start)) / 2
                else:
                    return min(x_right_start, y_right_start)

            elif x_left_end > y_right_start:
                r = m - 1
            else:
                l = m + 1
