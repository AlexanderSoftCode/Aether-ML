"""Naive pure-Python Philox4x32-10, the independent reference for the GPU dropout kernel."""
MASK32 = 0xffffffff


def philox4x32_10(counter, key):
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            (p1 >> 32) ^ c1 ^ k0,
            p1 & MASK32,
            (p0 >> 32) ^ c3 ^ k1,
            p0 & MASK32,
        )
        k0 = (k0 + 0x9E3779B9) & MASK32
        k1 = (k1 + 0xBB67AE85) & MASK32
    return c0, c1, c2, c3


def kept_elements(seed, offset, size, threshold):
    """Whether each of `size` flat elements is kept: 8 16-bit draws per Philox call, low half first."""
    kept = []
    for thread in range((size + 7) // 8):
        counter = (offset & MASK32, offset >> 32, thread & MASK32, thread >> 32)
        for word in philox4x32_10(counter, (seed & MASK32, seed >> 32)):
            kept += [(word & 0xffff) < threshold, (word >> 16) < threshold]
    return kept[:size]
