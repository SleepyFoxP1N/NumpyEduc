import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# 41. How to create a new column from exixting columns of a numpy array?
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = iris_2d[:, 0].astype('float')
petallength = iris_2d[:, 2].astype('float')
volume = (np.pi * petallength * (sepallength**2))/3
volume = volume[:, np.newaxis]
out = np.hstack([iris_2d, volume])

# 42. How to do probabilistic sampling in numpy?
species = iris[:, 4]
np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])
np.random.seed(100)
probs = np.r_[np.linspace(0, .500, num=50), np.linspace(.501, .750, num=50), np.linspace(.751, 1.0, num=50)]
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print(np.unique(species_out, return_counts=True))

# 43. How to get the second largest value of an array?
petal_len_setosa = iris[iris[:,4] == b'Iris-setosa', [2]].astype('float')
np.unique(np.sort(petal_len_setosa))[-2]