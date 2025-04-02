import numpy as np
from numba import jit

@jit(nopython=True)
def count_bits(x):
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

@jit(nopython=True)
def calculate_nonlinearity(sbox):
    max_bias = 0
    for i in range(256):
        for j in range(i + 1, 256):
            xor = sbox[i] ^ sbox[j]
            bias = count_bits(xor)
            max_bias = max(max_bias, bias)
    return 128 - (max_bias // 2)

@jit(nopython=True)
def calculate_avalanche(sbox):
    total_flips = 0
    for i in range(256):
        for j in range(8):
            flipped = i ^ (1 << j)
            diff = count_bits(sbox[i] ^ sbox[flipped])
            total_flips += diff
    return total_flips / (256 * 8)

@jit(nopython=True)
def calculate_differential_uniformity(sbox):
    freq = np.zeros((256, 256))
    for x in range(256):
        for y in range(256):
            diff = sbox[x] ^ sbox[y]
            freq[x][diff] += 1
    return freq.max()

@jit(nopython=True)
def calculate_bic(sbox):
    total_diff = 0
    for i in range(256):
        for j in range(256):
            diff = count_bits(sbox[i] ^ sbox[j])
            total_diff += diff
    return total_diff / (256 * 256)

def compute_metrics(sbox):
    """Calculate and return all metrics separately"""
    nonlinearity = calculate_nonlinearity(sbox)
    avalanche = calculate_avalanche(sbox)
    uniformity = calculate_differential_uniformity(sbox)
    bic = calculate_bic(sbox)
    
    return nonlinearity, avalanche, uniformity, bic

def fitness(sbox):
    nonlinearity, avalanche, uniformity, bic = compute_metrics(sbox)
    
    return (0.4 * nonlinearity) + (0.3 * avalanche) - (0.2 * uniformity) + (0.1 * bic)