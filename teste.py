
def teste(
    a, b, c
): 
    print(a)
    print(b)
    print(c)


b = teste, {'b': 20, 'a': 10, 'c': 30}

b[0](**b[1])