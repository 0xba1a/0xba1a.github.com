class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        h_map = {n: "nums1" for n in nums1}

        for n in nums2:
            if n in h_map and h_map[n] != "nums2":
                h_map[n] = "both"
            else:
                h_map[n] = "nums2"

        d_n1 = []
        d_n2 = []

        for n in h_map.keys():
            if h_map[n] == "nums1":
                d_n1.append(n)
            elif h_map[n] == "nums2":
                d_n2.append(n)

        return [d_n1, d_n2]
        