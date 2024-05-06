
def fuc(a):
    temp = [0, 0, 1, 1, 1, 3]
    for i in range(6, a+1):
        temp.append(temp[i-2] + temp[i-3] + temp[i-5])
    return temp[a]

print(fuc(100))
