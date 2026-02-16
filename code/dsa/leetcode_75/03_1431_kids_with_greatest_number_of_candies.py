class Solution:
    def find_max(self, candies):
        max_candies = 0
        for i in range(len(candies)):
            if candies[i] > max_candies:
                max_candies = candies[i]
        return max_candies

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_candies = self.find_max(candies)
        result = []

        for i in range(len(candies)):
            if candies[i] + extraCandies >= max_candies:
                result.append(True)
            else:
                result.append(False)
        return result