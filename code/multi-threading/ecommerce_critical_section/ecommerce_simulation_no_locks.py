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
RESET  = "\033[0m"

start_time = time.time()
def timestamp():
    return f"{time.time() - start_time:6.2f}s"

purchase_log = []

# ── Each user's thread runs this function ────────────────────────────

def process_user_activity(user, activities):
    for activity in activities:
        action = activity["action"]
        args   = activity["args"]
        duration = activity["time"]

        if action == "read_spec":
            phone = args[0]
            print(f"[{timestamp()}] {user}: Reading specs of {phone} -> {SPECS[phone]}...")
            time.sleep(duration)
            print(f"[{timestamp()}] {user}: Done: {phone} -> {SPECS[phone]}")

        elif action == "compare":
            phone_a, phone_b = args[0], args[1]
            print(f"[{timestamp()}] {user}: Comparing {phone_a} vs {phone_b}...")
            time.sleep(duration)
            print(f"[{timestamp()}] {user}: Done comparing {phone_a} vs {phone_b}")

        elif action == "buy":
            phone = args[0]

            # ── READ inventory ──
            stock = inventory[phone]
            print(f"{YELLOW}[{timestamp()}] {user}: Wants to buy {phone} (read stock = {stock}){RESET}")

            if stock <= 0:
                print(f"{YELLOW}[{timestamp()}] {user}: {phone} is out of stock!{RESET}")
                continue

            # Simulate checkout delay (payment, address, etc.)
            time.sleep(duration)

            # ── WRITE inventory (using the stale value we read earlier) ──
            inventory[phone] = stock - 1
            print(f"{YELLOW}[{timestamp()}] {user}: Purchased {phone}! (wrote stock = {stock - 1}){RESET}")
            purchase_log.append((user, phone))


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

    # PURCHASE LOGS
    print(f"\nSuccessful purchases ({len(purchase_log)} total):")
    for user, phone in purchase_log:
        print(f"  {user} bought {phone}")

    # INVENTORY VS SALES
    print(f"\nInitial inventory: {initial_inventory}")
    print(f"Final inventory:   {inventory}")

    # Calculate total sold per phone
    total_sold = {phone: 0 for phone in inventory}
    for _, phone in purchase_log:
        total_sold[phone] += 1
    print(f"\nPhones Sold:")
    for phone, sold in total_sold.items():
        print(f"  {phone}: {sold} sold")


    # print(f"\nUnits sold vs available:")
    # for phone in inventory:
    #     sold = total_sold[phone]
    #     available = initial_inventory[phone]
    #     status = " *** OVERSOLD ***" if sold > available else ""
    #     print(f"  {phone}: sold {sold} / {available} available{status}")

    run_time = time.time() - start_time
    print(f"\nTotal run time: {run_time:.2f}s")


if __name__ == "__main__":
    main()