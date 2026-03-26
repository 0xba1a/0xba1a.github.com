class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        max_alt = 0
        alt = 0

        for n in gain:
            alt += n
            if alt > max_alt:
                max_alt = alt

        return max_alt
        