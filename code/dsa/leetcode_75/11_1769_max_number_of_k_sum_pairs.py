class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        n_hash = {key: 0 for key in nums}
        ops = 0

        for n in nums:
            comp = k - n
            if comp in n_hash and n_hash[comp]:
                n_hash[comp] -= 1
                ops += 1
                continue

            n_hash[n] += 1

        return ops
        