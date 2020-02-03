# coding=utf-8
from tqdm import tqdm
import numpy as np
 
 
if __name__ == "__main__":
    tmp = np.arange(60000)
    for i in tqdm(range(1,60000),mininterval=1):
        print(tmp[i])

