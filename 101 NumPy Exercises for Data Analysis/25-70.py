import numpy as np

# 25. How to import a dataset with numbers and texts keeping the text intact in python numpy?
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
iris_3rows = iris[:3]

# 26. How to extract a particular column from 1D array of tuples?
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
species = np.array([row[4] for row in iris_1d])
particularCollumn = species[:5]

# 27. How to convert a 1d array of tuples to a 2d numpy array?
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
fourItems = iris_2d[:4]

# 28. How to compute the mean, median, standard deviation of a numpy array?
sepallength= np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
mu, med, std = np.mean(sepallength), np.median(sepallength), np.std(sepallength)

# 29. How to normalize an array so the values range exactly between 0 and 1?


print(mu, med, std)