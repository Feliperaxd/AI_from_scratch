a = [0, 0, 0, 0, 0]

for i, b in enumerate(range(10)):
    if i > len(a):
        a.append(0)
    a.insert(i, a[i]+b)
    
print(a)