import threading
import sys
import random

# (a + b)^2 = a^2 + b^2 + 2ab

def multiply(multiplicand, multiplier, key=None):
    # Simulate a CPU load wait time between 0.1 and 0.5 seconds
    wait_time = random.uniform(0.1, 0.5)
    threading.Event().wait(wait_time)

    global result
    product = multiplicand * multiplier
    result[key] = product
    print(f"Multiplication of {multiplicand} and {multiplier} is: {product}")


result = {}


def main():
    if len(sys.argv) < 3:
        print("Please provide two numbers as command line arguments.")
        sys.exit(1)

    a = int(sys.argv[1])
    b = int(sys.argv[2])

    a_square = threading.Thread(target=multiply, args=(a, a, "a_square"))
    b_square = threading.Thread(target=multiply, args=(b, b, "b_square"))
    ab_product = threading.Thread(target=multiply, args=(a, b, "ab_product"))

    a_square.start()
    b_square.start()
    ab_product.start()

    a_square.join()
    b_square.join()
    ab_product.join()

    a_plus_b_whole_square = result['a_square'] + result['b_square'] + 2 * result['ab_product']

    print(f"(a + b)^2 = {result['a_square']} + {result['b_square']} + 2 * {result['ab_product']} = {a_plus_b_whole_square}")

if __name__ == "__main__":
    main()
