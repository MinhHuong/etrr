def remove_spikes(a: list):
    for i in range(0, len(a) - 2):
        if a[i + 1] - a[i] == -1:
            if a[i + 2] - a[i + 1] == 1:
                a[i + 1] = 1
        elif a[i + 1] - a[i] == 1:
            if a[i + 2] - a[i + 1] == -1:
                a[i + 1] = 0
    return a
