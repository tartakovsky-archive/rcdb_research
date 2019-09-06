def pct_range(start, end, min_step, mult_step):
    last = start
    values = []
    while last < end:
        values.append(last)
        step = last * mult_step
        last += step if step > min_step else min_step

    if values and values[-1] != end:
        values.append(end)

    if all(isinstance(x, int) for x in [start, end, min_step, mult_step]):
        values = [int(value) for value in values]

    return values
