import math
import random

d = 0.01
n = random.randint(100,400)
t = 0

while True:
    print(d, math.sin(t))
    t += d / 100
    n -= 1

    if n < 0:
        d = random.uniform(0.01, 1.0)
        n = random.randint(4000, 6000)
