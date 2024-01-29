import numpy as np

a = np.array(range(1, 11))
size = 5

def split_X(dataset, size):
    aaa=[]
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    

   
  
bbb = split_X(a, size)
# print(bbb)          # [[ 1  2  3  4  5][ 2  3  4  5  6][ 3  4  5  6  7][ 4  5  6  7  8][ 5  6  7  8  9][ 6  7  8  9 10]]
# print(bbb.shape)    # (6, 5)

X = bbb[:, :-1]       
y = bbb[:, -1]
print(X,y)                         
print(X.shape, y.shape)             


