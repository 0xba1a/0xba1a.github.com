import sys

def main(name):
    print("I am main function")
    greet(name)

def greet(name):
    print(f"Hello, {name}!")

if __name__ == "__main__":
    main(sys.argv[1])