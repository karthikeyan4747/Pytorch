import sklearn
from sklearn.datasets import make_circles

n_samples = 1000


x,y = make_circles(n_samples,noise=0.03,random_state=42)

print(len(x),len(y))

import pandas as pd