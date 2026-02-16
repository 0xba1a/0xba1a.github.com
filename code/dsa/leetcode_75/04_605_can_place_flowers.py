class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if not n:
            return True

        f_len = len(flowerbed)

        if f_len == 1 and not flowerbed[0]:
            return n == 1

        if f_len > 1 and not flowerbed[0] and not flowerbed[1]:
            n -= 1
            flowerbed[0] = 1

        i = 2
        while i < f_len-1 and n:
            if not flowerbed[i] and not flowerbed[i-1] and not flowerbed[i+1]:
                n -= 1
                flowerbed[i] = 1
            i += 1

        if n and f_len > 1 and not flowerbed[-1] and not flowerbed[-2]:
            n -= 1
            flowerbed[-1] = 1
        
        return n == 0