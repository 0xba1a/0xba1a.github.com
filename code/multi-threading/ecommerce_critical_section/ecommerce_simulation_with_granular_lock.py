import json
import threading
import time
from pathlib import Path

inventory = {
    "Pixel 10 Pro": 2,
    "iPhone 17 Pro": 2,
    "Samsung S25 Ultra": 3,
}

SPECS = {
    "Pixel 10 Pro": "6.3\" OLED, Tensor G5, 16GB RAM, 256GB, 50MP camera, Android 16",
    "iPhone 17 Pro": "6.3\" OLED, A19 Pro, 12GB RAM, 256GB, 48MP camera, iOS 19",
    "Samsung S25 Ultra": "6.9\" AMOLED, Snapdragon 8 Elite, 16GB RAM, 256GB, 200MP camera, Android 16",
}

# ── ANSI colors ──────────────────────────────────────────────────────
YELLOW = "\033[38;2;255;200;0m"
RED    = "\033[38;2;255;60;60m"
RESET  = "\033[0m"

# ── Per-phone colors (to show interleaving in the log) ───────────────
PHONE_COLOR = {
    "Pixel 10 Pro":      "\033[38;2;100;255;100m",  # Bright Green
    "iPhone 17 Pro":     "\033[38;2;220;120;255m",  # Purple
    "Samsung S25 Ultra": "\033[38;2;255;140;0m",    # Orange
}

start_time = time.time()
purchase_log = []

# ── One lock per product (granular locking) ──────────────────────────
inventory_locks = {}
for phone in inventory:
    inventory_locks[phone] = threading.Lock()

# ── Per-product wait time (each protected by its own inventory lock) ─
wait_times = {phone: 0.0 for phone in inventory}


def timestamp():
    return f"{time.time() - start_time:6.2f}s"


# ── Each user's thread runs this function ────────────────────────────

def process_user_activity(user, activities):
    for activity in activities:
        action = activity["action"]
        args   = activity["args"]
        duration = activity["time"]

        if action == "read_spec":
            phone = args[0]
            print(f"[{timestamp()}] {user}: Reading specs of {phone}...")
            time.sleep(duration)
            print(f"[{timestamp()}] {user}: Done: {phone} -> {SPECS[phone]}")

        elif action == "compare":
            phone_a, phone_b = args[0], args[1]
            print(f"[{timestamp()}] {user}: Comparing {phone_a} vs {phone_b}...")
            time.sleep(duration)
            print(f"[{timestamp()}] {user}: Done comparing {phone_a} vs {phone_b}")

        elif action == "buy":
            phone = args[0]

            # ── ACQUIRE per-product lock before reading inventory ──
            wait_start = time.time()
            print(f"[{timestamp()}] {user}: Waiting for {phone} lock...")

            lock = inventory_locks[phone]
            lock.acquire()

            wait_duration = time.time() - wait_start

            wait_times[phone] += wait_duration

            print(f"[{timestamp()}] {user}: Lock acquired (waited {wait_duration:.2f}s)")

            # ── READ inventory (safe — only this product is locked) ──
            stock = inventory[phone]
            c = PHONE_COLOR[phone]
            print(f"{c}[{timestamp()}] {user}: Wants to buy {phone} (read stock = {stock}){RESET}")

            if stock <= 0:
                print(f"{c}[{timestamp()}] {user}: {phone} is out of stock!{RESET}")
                lock.release()
                continue

            # Simulate checkout delay (payment, address, etc.)
            time.sleep(duration)

            # ── WRITE inventory ──
            inventory[phone] = stock - 1
            print(f"{c}[{timestamp()}] {user}: Purchased {phone}! (wrote stock = {stock - 1}){RESET}")
            purchase_log.append((user, phone))

            # ── RELEASE per-product lock ──
            lock.release()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # Load user activity log
    with open(f"{Path(__file__).parent}/ecommerce_simulation.json") as f:
        data = json.load(f)

    initial_inventory = dict(inventory)  # snapshot before threads start

    # Create one thread per user
    threads = []
    for user, activities in data["users"].items():
        t = threading.Thread(target=process_user_activity, args=(user, activities))
        threads.append(t)

    print(f"Initial inventory: {initial_inventory}")
    print(f"Starting {len(threads)} user threads...\n")

    input("Press Enter to start the simulation...")

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # ── Print results ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Final inventory:   {inventory}")
    print(f"Initial inventory: {initial_inventory}")

    total_sold = {phone: 0 for phone in inventory}
    for _, phone in purchase_log:
        total_sold[phone] += 1

    print(f"\nSuccessful purchases ({len(purchase_log)} total):")
    for user, phone in purchase_log:
        print(f"  {user} bought {phone}")

    print(f"\nUnits sold vs available:")
    for phone in inventory:
        sold = total_sold[phone]
        available = initial_inventory[phone]
        status = " *** OVERSOLD ***" if sold > available else ""
        print(f"  {phone}: sold {sold} / {available} available{status}")

    print(f"\nLock wait time per product:")
    for phone, wt in wait_times.items():
        print(f"  {phone}: {wt:.2f}s")
    total_wait = sum(wait_times.values())
    print(f"\nTotal contention across all products: {total_wait:.2f}s")

    run_time = time.time() - start_time
    print(f"Total run time: {run_time:.2f}s")


if __name__ == "__main__":
    main()
