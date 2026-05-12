import threading
import sys
import random

# (a + b)^2 = a^2 + b^2 + 2ab

def multiply(a, b, result_dict=None, key=None):
    # add a random wait between 1 to 5 seconds
    wait_time = random.randint(1, 5)
    print(f"Thread for multiplying {a} and {b} will wait for {wait_time} seconds.")
    threading.Event().wait(wait_time)

    result = a * b
    if result_dict is not None and key is not None:
        result_dict[key] = result
    print(f"Multiplication of {a} and {b} is: {result}")

result = {
    "a_square": 0,
    "b_square": 0,
    "ab_product": 0
}

def main():
    if len(sys.argv) < 3:
        print("Please provide two numbers as command line arguments.")
        sys.exit(1)

    a = int(sys.argv[1])
    b = int(sys.argv[2])

    a_square = threading.Thread(target=multiply, args=(a, a, result, "a_square"))
    b_square = threading.Thread(target=multiply, args=(b, b, result, "b_square"))
    ab_product = threading.Thread(target=multiply, args=(a, b, result, "ab_product"))

    a_square.start()
    b_square.start()
    ab_product.start()

    a_square.join()
    b_square.join()
    ab_product.join()

    print(f"(a + b)^2 = {result['a_square']} + {result['b_square']} + 2 * {result['ab_product']} = {result['a_square'] + result['b_square'] + 2 * result['ab_product']}")

if __name__ == "__main__":
    main()