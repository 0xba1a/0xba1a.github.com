import sys

def factorial(n):
    if n == 1:
        print("1! = 1")
        return 1
    else:
        print(f"{n} x {n-1}!")
        return n * factorial(n - 1)

if __name__ == "__main__":
    num = int(sys.argv[1])
    print(f"Factorial of {num} is {factorial(num)}")