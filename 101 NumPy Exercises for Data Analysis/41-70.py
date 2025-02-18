import numpy as np
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime

np.set_printoptions(precision=3)

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype=object, encoding='utf-8')

# 41. Create a new column from existing columns and normalize it
iris_2d = iris[:, :4].astype(float)
sepallength, petallength = iris_2d[:, 0], iris_2d[:, 2]
volume = (np.pi * petallength * (sepallength**2)) / 3
volume = (volume - volume.min()) / volume.ptp()  # Normalize volume
iris_extended = np.hstack([iris_2d, volume[:, None]])

# 42. Perform stratified sampling based on species
np.random.seed(100)
species = iris[:, 4].astype(str)
stratified_samples = np.hstack([
    np.random.choice(iris[species == s], size=5, replace=False) for s in np.unique(species)
])

# 43. Find the second largest petal length for each species
unique_species = np.unique(species)
second_largest_values = {
    s: np.unique(np.sort(iris[species == s, 2].astype(float)))[-2]
    for s in unique_species
}

# 44. Sort a 2D array by two columns (ascending and descending order)
sorted_iris = iris[np.lexsort((-iris[:, 2].astype(float), iris[:, 0].astype(float)))]

# 45. Find the most frequent petal width
most_frequent = np.bincount(iris[:, 3].astype(float).astype(int)).argmax()

# 46. Find the first occurrence of a value greater than the 90th percentile
threshold = np.percentile(iris[:, 3].astype(float), 90)
first_above_threshold = np.argmax(iris[:, 3].astype(float) > threshold)

# 47. Replace extreme outliers (IQR method) with the median
Q1, Q3 = np.percentile(iris_2d, [25, 75], axis=0)
iqr_range = Q3 - Q1
outliers = (iris_2d < Q1 - 1.5 * iqr_range) | (iris_2d > Q3 + 1.5 * iqr_range)
iris_no_outliers = np.where(outliers, np.median(iris_2d, axis=0), iris_2d)

# 48. Get top n values from each column
top_5_values_per_column = np.sort(iris_2d, axis=0)[-5:, :]

# 49. Row-wise count of unique values
row_value_counts = [np.unique(row, return_counts=True) for row in iris_2d.astype(int)]

# 50. Flatten a 2D array without using `.flatten()`
flat_array = iris_2d.ravel()

# 51. One-hot encoding using NumPy for species column
species_unique = np.unique(species)
one_hot_encoded = (species[:, None] == species_unique).astype(int)

# 52. Create indices representing group IDs based on species
group_ids = np.searchsorted(species_unique, species)

# 53. Compute rank for each value in a NumPy array
np.random.seed(10)
a = np.random.randint(20, size=20)
ranks = a.argsort().argsort()

# 54. Compute rank for values in a 2D array row-wise
a_2d = np.random.randint(1, 20, (3, 5))
rowwise_ranks = np.argsort(a_2d, axis=1).argsort(axis=1)

# 55. Compute moving average with a dynamic window size
window_sizes = np.random.randint(2, 5, size=10)
moving_averages = np.array([np.convolve(a, np.ones(w)/w, mode='valid') for a, w in zip(a_2d, window_sizes)])

# 56. Find the maximum value in each row without using `.max()`
max_per_row = np.apply_along_axis(lambda x: np.sort(x)[-1], axis=1, arr=a_2d)

# 57. Compute min-max ratio per row
min_max_ratio = np.min(a_2d, axis=1) / np.max(a_2d, axis=1)

# 58. Detect duplicate rows in a 2D NumPy array
_, unique_indices = np.unique(iris_2d, axis=0, return_index=True)
duplicate_indices = np.setdiff1d(np.arange(len(iris_2d)), unique_indices)

# 59. Compute the grouped mean per species for each numerical column
grouped_means = {s: iris_2d[species == s].mean(axis=0) for s in np.unique(species)}

# 60. Convert a PIL image to grayscale NumPy array
from PIL import Image
import requests
from io import BytesIO
image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
image = Image.open(BytesIO(requests.get(image_url).content)).convert('L').resize((150, 150))
image_arr = np.asarray(image)

# 61. Remove NaN values while keeping shape intact (replace with mean)
a = np.array([1, 2, 3, np.nan, 5, 6, 7, np.nan])
a[np.isnan(a)] = np.nanmean(a)

# 62. Compute cosine similarity between two random vectors
a, b = np.random.rand(10), np.random.rand(10)
cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 63. Identify peaks in a time-series dataset
time_series = np.random.randint(0, 10, 20)
peaks = np.where((time_series[1:-1] > time_series[:-2]) & (time_series[1:-1] > time_series[2:]))[0] + 1

# 64. Subtract the median from each row in a 2D array
median_centered = iris_2d - np.median(iris_2d, axis=1, keepdims=True)

# 65. Find the index of the nth occurrence of a value
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
n, value = 5, 1
nth_occurrence = np.where(x == value)[0][n - 1] if np.count_nonzero(x == value) >= n else -1

# 66. Convert NumPy datetime64 array to Python datetime list
from datetime import datetime
dates = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
python_dates = dates.astype(datetime)

# 67. Compute exponentially weighted moving average (EWMA)
def ewma(arr, alpha=0.3):
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

ewma_values = ewma(time_series)

# 68. Generate a Fibonacci sequence using NumPy
def fibonacci(n):
    fib_seq = np.zeros(n)
    fib_seq[:2] = 1
    for i in range(2, n):
        fib_seq[i] = fib_seq[i - 1] + fib_seq[i - 2]
    return fib_seq

fib_numbers = fibonacci(10)

# 69. Fill in missing dates in an irregular time-series
dates = np.array(['2023-01-01', '2023-01-04', '2023-01-06'], dtype='datetime64[D]')
all_dates = np.arange(dates.min(), dates.max() + 1, dtype='datetime64[D]')

# 70. Generate sliding windows over a sequence
arr = np.arange(15)
stride, window = 2, 4
sliding_windows = np.array([arr[i:i+window] for i in range(0, len(arr) - window + 1, stride)])
