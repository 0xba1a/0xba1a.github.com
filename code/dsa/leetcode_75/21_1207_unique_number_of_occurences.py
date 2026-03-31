class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        h_map = {}

        for n in arr:
            if n in h_map:
                h_map[n] += 1
            else:
                h_map[n] = 1
            
        occr_set = set()

        for n in h_map.keys():
            occurance = h_map[n]
            if occurance in occr_set:
                return False
            occr_set.add(occurance)

        return True
        