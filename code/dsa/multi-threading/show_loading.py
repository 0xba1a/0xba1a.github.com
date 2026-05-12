import time
import sys
from threading import Thread

LOADING = False

def show_loading():
    global LOADING
    while LOADING:
        for char in "|/-\\":
            print(f"\rLoading... {char}", end="")
            time.sleep(0.1)

    print("\rLoading... Done!   ")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a duration in seconds as a command line argument.")
        sys.exit(1)

    duration = int(sys.argv[1])

    LOADING = True
    loading_thread = Thread(target=show_loading)
    loading_thread.start()

    time.sleep(duration)  # Simulate some work being done for the specified duration

    LOADING = False
    loading_thread.join()

    # Simulate some work being done
    print("\nDone!")