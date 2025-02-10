import numpy as np
np.set_printoptions(precision=3)

# 25. How to import a dataset with numbers and texts keeping the text intact in python numpy?
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object', encoding='utf-8')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
iris_3rows = iris[:3]

# 26. How to extract a particular column from 1D array of tuples?
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding='utf-8')
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
sepalMax, sepalMin = np.max(sepallength), np.min(sepallength)
sepalMax, sepalMin = sepallength.max(), sepallength.min()
sepallength_normalized = (sepallength - sepalMin)/(sepalMax - sepalMin)
sepallength_normalized = (sepallength - sepalMin)/sepallength.ptp()

# 30. How to compute the softmax score?
sepallength = np.array([float(row[0]) for row in iris_1d])
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 31. How to find the percentile scores of a numpy array?
percentile_scores = np.percentile(sepallength, [5, 95])

# 32. How to insert values at random positions in an array?
np.random.seed(777)
i, j = np.where(iris_2d)
iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# 33. How to find the position of missing values in a numpy array?
number_missing = np.sum(iris_2d[:, 0].sum())
missing_values = np.where(np.isnan(iris_2d[:, 0]))

# 34. How to filter a numpy array based on two or more conditions?
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
iris_2d_filtered = iris_2d[condition]

# 35. How to drop rows that contain a missing value form a numpy array?
any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
iris_2d_filtered = iris_2d[any_nan_in_row][:5]
iris_2d_filtered = iris_2d[np.sum(np.isnan(iris_2d), axis=1) == 0][:5]

# 36. How to find the correlation between two coluns of a numpy array?
np.corrcoef(iris[:, 0], iris[:, 2])[0, 1]
from scipy.stats.stats import pearsonr  
corr, p_value = pearsonr(iris[:, 0], iris[:, 2])

# 37. How to find if a given array has any null values?
hasNull_value = np.isnan(iris_2d).any()

# 38. How to replace all missing values with 0 in a numpy array?
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float, usecols=[0,1,2,3]')
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
iris_2d[np.isnan(iris_2d)] = 0
replaced_nan_with_0 = iris_2d[:4]

# 39. How to find the count of unique values in a numpy array?
species = np.array([row.tolist()[4] for row in iris])
unique_values = np.unique(species, return_counts=True)

# 40. How to convert a numeric to a categorical (text) array?
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]

