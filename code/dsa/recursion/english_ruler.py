import sys

def draw_line(nticks, label=""):
    print("-" * nticks + " " + label)

def draw_interval(nticks):
    if nticks > 0:
        draw_interval(nticks - 1)
        draw_line(nticks)
        draw_interval(nticks - 1)

def draw_ruler(nticks, len):
    draw_line(len, "0")
    for i in range(1, len + 1):
        draw_interval(nticks - 1)
        draw_line(len, str(i))

if __name__ == "__main__":
    draw_ruler(int(sys.argv[1]), int(sys.argv[2]))