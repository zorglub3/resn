from numpy.random import normal

mem = [0, 0]
y = 0

while True:
    z = normal(loc = 0, scale = 0.001, size = None)
    y = 1 - 1.4 * mem[0] * mem[0] + 0.3 * mem[1] + z
    print(mem[0], y)
    mem = [y, mem[0]]
