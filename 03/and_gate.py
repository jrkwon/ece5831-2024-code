def and_gate(x1, x2):
    w1 = 1
    w2 = 1
    th = 1.9
    if (x1*w1 + x2*w2) > th:
        return 1
    else:
        return 0

if __name__ == "__main__":
    y = and_gate(1, 0)
    print(y)
    print("this is called from command line.")