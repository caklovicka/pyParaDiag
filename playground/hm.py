def f(a):
    a = a.copy()
    a[0] += 1
    a[1] += 2

    return a

def g(x):
    return x+2

x = [0, 0]
f(x)
print(f(x), x)

z = 3
print(z, g(z))