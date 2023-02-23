from math import floor, ceil


def round_up(float):
    if float - floor(float) == 0.5:
        return ceil(float)
    else:
        return round(float)
