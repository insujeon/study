import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def pprint(arr):
    print("type:{}".format(type(arr)))
    print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array's Data:\n", arr)

data = [1,2,3,4,5]
arr = np.array(data)
arr.shape

data = [[1,2,3,4],[5,6,7,8]]
arr = np.array(data)
arr.shape
arr.ndim

np.empty((3,3))
np.ones((3,3))
np.zeros((3,3))
np.eye(3,3)

np.full((3,3),7)

np.arange(1,10,2)
np.arange(15)

arr = np.array([1,2,3,4,5])
arr.dtype

float_arr = arr.astype(np.float64)
float_arr.dtype  
float_arr

arr = float_arr.astype(np.int64)
arr.dtype

arr = pd.DataFrame(np.random.rand(50,100))

arr.head()
arr.tail()

print(arr)

arr = np.array(range(9))
arr = np.arange(9)
arr = arr.reshape(3,3)
arr.shape
arr

arr[0][2]
arr[0,2]

arr[arr>3]=3
arr

arr.T
np.sqrt(arr)
np.square(arr)
arr**2

pprint(arr)
arr

np.ones_like(arr)







