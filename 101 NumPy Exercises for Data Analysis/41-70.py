import numpy as np
from scipy.stats import pearsonr
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime

np.set_printoptions(precision=3)

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype=object, encoding='utf-8')

# 41. Create a new column from existing columns
iris_2d = iris[:, :4].astype(float)
sepallength, petallength = iris_2d[:, 0], iris_2d[:, 2]
volume = (np.pi * petallength * (sepallength**2)) / 3
out = np.hstack([iris_2d, volume[:, None]])

# 42. Probabilistic sampling
np.random.seed(100)
species = iris[:, 4].astype(str)
probs = np.linspace(0, 1, num=150)
species_out = species[np.searchsorted(probs, np.random.random(150))]
unique_species_counts = np.unique(species_out, return_counts=True)

# 43. Second largest value in an array
petal_len_setosa = iris[iris[:, 4] == b'Iris-setosa', 2].astype(float)
second_largest = np.unique(np.sort(petal_len_setosa))[-2]

# 44. Sort a 2D array by a column
sorted_iris = iris[np.argsort(iris[:, 0])][:20]

# 45. Find the most frequent value in an array
vals, counts = np.unique(iris[:, 2], return_counts=True)
most_frequent = vals[np.argmax(counts)]

# 46. Position of first occurrence of a value greater than given value
first_occurrence = np.argmax(iris[:, 3].astype(float) > 1.0)

# 47. Replace all values greater than a given value with a cutoff
np.random.seed(777)
a = np.random.uniform(1, 50, 20)
clipped_a = np.clip(a, 10, 30)

# 48. Get positions of top n values
top_5_positions = np.argpartition(-a, 5)[:5]
top_5_values = a[top_5_positions]

# 49. Row-wise counts of all possible values in an array
np.random.seed(100)
arr = np.random.randint(1, 11, size=(6, 10))
row_value_counts = [np.unique(row, return_counts=True) for row in arr]

# 50. Convert an array of arrays into a flat 1D array
arr1, arr2, arr3 = np.arange(3), np.arange(3, 7), np.arange(7, 10)
flat_arr = np.concatenate([arr1, arr2, arr3])

# 51. Generate one-hot encodings
np.random.seed(101)
arr = np.random.randint(1, 4, size=6)
one_hot = (arr[:, None] == np.unique(arr)).astype(int)

# 52. Create row numbers grouped by a categorical variable
species_small = np.sort(np.random.choice(species, size=20))

# 53. Create group IDs based on a categorical variable
group_ids = {val: i for i, val in enumerate(np.unique(species_small))}
species_grouped = np.array([group_ids[val] for val in species_small])

# 54. Rank items in an array
np.random.seed(10)
a = np.random.randint(20, size=20)
ranks = np.argsort(a).argsort()

# 55. Rank items in a multidimensional array
a = np.random.randint(20, size=(2, 5))
ranked_multi = a.ravel().argsort().argsort().reshape(a.shape)

# 56. Find the maximum value in each row
a = np.random.randint(1, 10, (5, 3))
max_per_row = np.max(a, axis=1)

# 57. Compute min-by-max for each row
min_by_max = np.min(a, axis=1) / np.max(a, axis=1)

# 58. Find duplicate records
a = np.random.randint(0, 5, 10)
unique_positions = np.unique(a, return_index=True)[1]
duplicates = np.setdiff1d(np.arange(len(a)), unique_positions)

# 59. Find grouped mean
numeric_col, group_col = iris[:, 1].astype(float), iris[:, 4]
grouped_means = {group: numeric_col[group_col == group].mean() for group in np.unique(group_col)}

# 60. Convert a PIL image to a NumPy array
image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
image = Image.open(BytesIO(requests.get(image_url).content)).resize((150, 150))
image_arr = np.asarray(image)

# 61. Drop all missing values
a = np.array([1, 2, 3, np.nan, 5, 6, 7, np.nan])
clean_a = a[~np.isnan(a)]

# 62. Compute Euclidean distance between two arrays
a, b = np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 7, 8])
euclidean_dist = np.linalg.norm(a - b)

# 63. Find local maxima in a 1D array
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
local_maxima = np.where(np.r_[False, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], False])[0]

# 64. Subtract a 1D array from a 2D array row-wise
a_2d = np.array([[3, 3, 3], [4, 4, 4], [5, 5, 5]])
b_1d = np.array([1, 2, 3])
rowwise_subtraction = a_2d - b_1d[:, None]

# 65. Find the index of the nth occurrence of an item
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
n = 5
nth_occurrence = np.where(x == 1)[0][n-1]

# 66. Convert NumPy datetime64 to datetime object
dt64 = np.datetime64('2018-02-25 22:10:10')
dt_object = dt64.astype(datetime)

# 67. Compute moving average
Z = np.random.randint(10, size=10)
moving_avg = np.convolve(Z, np.ones(3)/3, mode='valid')

# 68. Create a NumPy sequence given start, length, step
length, start, step = 10, 5, 3
sequence = np.arange(start, start + (step * length), step)

# 69. Fill in missing dates in an irregular series
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
filled_dates = np.hstack([np.arange(dates[i], dates[i + 1]) for i in range(len(dates) - 1)])
full_dates = np.hstack([filled_dates, dates[-1]])

# 70. Create strides from a 1D array
arr = np.arange(15)
stride_len, window_len = 2, 4
num_strides = ((arr.size - window_len) // stride_len) + 1
strided_array = np.array([arr[i:i+window_len] for i in range(0, num_strides*stride_len, stride_len)])

