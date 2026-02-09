class Solution1:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s

        rows = numRows
        cols = len(s)
        matrix = [['-' for _ in range(cols)] for _ in range(rows)]

        row = col = 0
        zig = True

        for char in s:
            matrix[row][col] = char

            if zig == True:
                row += 1

                if row >= numRows:
                    row = numRows - 2
                    col += 1
                    zig = False

            else:
                matrix[row][col] = char

                row -= 1
                col += 1

                if row < 0:
                    row = 1
                    col -= 1
                    zig = True
                elif row == 0:
                    row = 0
                    zig = True

        final_str = ''
        for i in range(rows):
            for j in range(cols):
                final_str += matrix[i][j] if matrix[i][j] != '-' else ''

        print(final_str)
        return final_str

class Solution():
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s

        str_matrix = ["" for _ in range(numRows)]

        row = 0
        direction = 1

        for c in s:
            str_matrix[row] += c
            row += direction

            if row == 0 or row == numRows-1:
                direction *= -1

        return "".join(str_matrix)

s = "PAYPALISHIRING"
numRows = 3
# s = "A"
# numRows  = 1
# s = "ABCD"
# numRows = 2
sol = Solution()
output_str = sol.convert(s, numRows)
print(output_str)