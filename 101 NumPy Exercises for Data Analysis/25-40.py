import numpy as np
from scipy.stats import pearsonr

np.set_printoptions(precision=3)

# 25. Import dataset with numbers and text intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype=object, encoding='utf-8')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
iris_3rows = iris[:3]

# 26. Extract a particular column from a 1D array of tuples
species = iris[:, 4].astype(str)  # Direct extraction instead of list comprehension
particular_column = species[:5]

# 27. Convert a 1D array of tuples to a 2D numpy array
iris_2d = np.genfromtxt(url, delimiter=',', dtype=float, usecols=[0,1,2,3])
four_items = iris_2d[:4]

# 28. Compute mean, median, standard deviation
sepallength = iris_2d[:, 0]
mu, med, std = np.mean(sepallength), np.median(sepallength), np.std(sepallength)

# 29. Normalize an array to range [0,1]
sepallength_normalized = (sepallength - sepallength.min()) / sepallength.ptp()

# 30. Compute the softmax score
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

softmax_scores = softmax(sepallength)

# 31. Find the percentile scores of a numpy array
percentile_scores = np.percentile(sepallength, [5, 95])

# 32. Insert NaN values at random positions in an array
np.random.seed(777)
random_rows = np.random.randint(iris_2d.shape[0], size=20)
random_cols = np.random.randint(iris_2d.shape[1], size=20)
iris_2d[random_rows, random_cols] = np.nan  # Efficient random NaN insertion

# 33. Find the position of missing values in a numpy array
missing_values = np.where(np.isnan(iris_2d))
number_missing = np.sum(np.isnan(iris_2d))

# 34. Filter a numpy array based on two or more conditions
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
iris_2d_filtered = iris_2d[condition]

# 35. Drop rows that contain missing values
iris_2d_filtered = iris_2d[~np.isnan(iris_2d).any(axis=1)]

# 36. Find the correlation between two columns of a numpy array
correlation = np.corrcoef(iris_2d[:, 0], iris_2d[:, 2])[0, 1]
correlation, p_value = pearsonr(iris_2d[:, 0], iris_2d[:, 2])

# 37. Check if a given array has any null values
has_null_value = np.isnan(iris_2d).any()

# 38. Replace all missing values with 0 in a numpy array
iris_2d[np.isnan(iris_2d)] = 0
replaced_nan_with_0 = iris_2d[:4]

# 39. Find the count of unique values in a numpy array
species = iris[:, 4].astype(str)
unique_values = np.unique(species, return_counts=True)

# 40. Convert numeric values to categorical (text) array
petal_length_bin = np.digitize(iris_2d[:, 2], [0, 3, 5, 10])
label_map = {1: 'small', 2: 'medium', 3: 'large'}
petal_length_cat = np.array([label_map.get(x, 'unknown') for x in petal_length_bin])
