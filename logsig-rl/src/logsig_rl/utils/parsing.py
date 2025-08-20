def parse_range(spec: str):
    """Parse 'start:end:step' into a list of floats inclusive of end (with tolerance)."""
    a, b, c = map(float, spec.split(":"))
    vals = []
    x = a
    # avoid float accumulation issues
    k = 0
    while x <= b + 1e-9 and k < 10_000:
        vals.append(round(x, 12))
        x = a + (k + 1) * c
        k += 1
    return vals
