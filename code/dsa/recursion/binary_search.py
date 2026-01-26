def binary_search(arr, target, start, end):
    if start > end:
        return -1  # Target not found

    mid = (start + end) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, end)
    else:
        return binary_search(arr, target, start, mid - 1)
    

# Example usage:
if __name__ == "__main__":
    array = [2, 4, 5, 7, 9, 12, 14, 17, 19, 22, 25, 27, 28, 33, 37]
    target_value = 22
    result = binary_search(array, target_value, 0, len(array) - 1)
    if result != -1:
        print(f"Element found at index: {result}")
    else:
        print(f"Element not found in the array at index: {result}.")