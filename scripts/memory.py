from random import uniform

mem = [0, 0, 0, 0, 0]
x = 0
y = 0

while True:
    x = uniform(0, 1) * 2 - 1
    y, *mem = mem
    print(x, y)
    mem = *mem, x
