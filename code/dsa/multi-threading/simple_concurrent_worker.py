import sys
import random
import threading
from concurrent.futures import ThreadPoolExecutor

def multiply(a, b):
    # add a random wait between 1 to 5 seconds
    wait_time = random.randint(1, 5)
    print(f"Thread for multiplying {a} and {b} will wait for {wait_time} seconds.")
    threading.Event().wait(wait_time)

    result = a * b
    print(f"Multiplication of {a} and {b} is: {result}")
    return result

def main():
    if len(sys.argv) < 3:
        print("Please provide two numbers as command line arguments.")
        sys.exit(1)

    a = int(sys.argv[1])
    b = int(sys.argv[2])

    with ThreadPoolExecutor(max_workers=2) as executor:
        a_square_work = executor.submit(multiply, a, a)
        b_square_work = executor.submit(multiply, b, b)
        ab_product_work = executor.submit(multiply, a, b)

        a_square = a_square_work.result()
        b_square = b_square_work.result()
        ab_product = ab_product_work.result()

        print(f"(a + b)^2 = {a_square} + {b_square} + 2 * {ab_product} = {a_square + b_square + 2 * ab_product}")

if __name__ == "__main__":
    main()