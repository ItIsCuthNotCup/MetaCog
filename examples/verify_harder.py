"""Brute-force ground truth for the HARDER problem set in live_demo.py."""

from itertools import combinations, permutations, product
from math import comb, gcd, isqrt


def is_prime(n):
    return n > 1 and all(n % p for p in range(2, isqrt(n) + 1))


# 1. n <= 1000 such that n^2+n+1 is prime
p1 = sum(1 for n in range(1, 1001) if is_prime(n * n + n + 1))

# 2. number of ordered pairs (a,b), 1<=a,b<=100, with gcd(a,b)=1 and a+b divisible by 7
p2 = sum(1 for a in range(1, 101) for b in range(1, 101) if gcd(a, b) == 1 and (a + b) % 7 == 0)

# 3. 6-digit numbers with strictly increasing digits and digit sum divisible by 3
p3 = sum(1 for c in combinations(range(1, 10), 6) if sum(c) % 3 == 0)

# 4. number of integers 1..2024 expressible as difference of two squares of nonneg integers
p4 = sum(1 for n in range(1, 2025) if n % 4 != 2)

# 5. derangements of 1..8 with p(1) = 2
p5 = sum(1 for p in permutations(range(8)) if all(p[i] != i for i in range(8)) and p[0] == 1)


# 6. number of binary strings of length 12 with no three consecutive equal bits
def no_triple(s):
    return all(not (s[i] == s[i + 1] == s[i + 2]) for i in range(len(s) - 2))


p6 = sum(1 for s in product("01", repeat=12) if no_triple(s))

# 7. (unused) sum of n <= 1000 with n | 2^n - 2
p7 = sum(n for n in range(1, 1001) if pow(2, n, n) == 2 % n)

# 8. number of lattice paths from (0,0) to (8,8) with unit steps right/up that never touch (4,4)
p8 = comb(16, 8) - comb(8, 4) ** 2


# 9. number of ordered triples (a,b,c) of positive integers with abc = 720
def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


p9 = sum(1 for a in divisors(720) for b in divisors(720 // a))


# 10. n <= 500 for which n! has exactly 3 more trailing zeros than (n-1)!
def tz(n):
    z = 0
    k = 5
    while k <= n:
        z += n // k
        k *= 5
    return z


p10 = sum(1 for n in range(1, 501) if tz(n) - tz(n - 1) == 3)

# 11. how many 4x4 0/1 matrices have every row and column sum even
p11 = sum(
    1
    for m in product((0, 1), repeat=16)
    if all(sum(m[4 * r : 4 * r + 4]) % 2 == 0 for r in range(4))
    and all(sum(m[c::4]) % 2 == 0 for c in range(4))
)

# 12. sum of digits of 3^100
p12 = sum(map(int, str(3**100)))

# 13. number of integers 1..10000 with digit sum 20
p13 = sum(1 for n in range(1, 10001) if sum(map(int, str(n))) == 20)

# 14. number of subsets of {1..12} whose sum is divisible by 12 (including empty set)
p14 = sum(1 for mask in range(1 << 12) if sum(i + 1 for i in range(12) if mask >> i & 1) % 12 == 0)

# 15. number of primes p<1000 such that p+2 and p+6 are also prime
p15 = sum(1 for p in range(2, 1000) if is_prime(p) and is_prime(p + 2) and is_prime(p + 6))

# 16. last three digits of 3^2024
p16 = pow(3, 2024, 1000)

for i, v in enumerate([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16], 1):
    print(i, v)
