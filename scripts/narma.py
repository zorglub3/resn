from random import uniform

n = 10
alpha = 0.3
beta = 0.05
gamma = 1.5
delta = 0.1

y_mem = [0] * n
u_mem = [0] * n

while True:
    u = uniform(0, 0.5)
    y = alpha * y_mem[0] + beta * y_mem[0] * sum(y_mem) + gamma * u_mem[n-1] * u_mem[0] + delta
    print(y_mem[0], y)
    y_mem = y, *y_mem[:n-1]
    u_mem = u, *u_mem[:n-1]
