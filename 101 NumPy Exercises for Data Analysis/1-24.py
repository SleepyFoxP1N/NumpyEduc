import numpy as np
import sys

# 1. Print NumPy version
import numpy as np
print(np.__version__)

# 2. Create a 1D array of even numbers from 0 to 18
array_1d = np.arange(0, 20, 2)

# 3. Create a 3x3 identity matrix
identity_matrix = np.eye(3, dtype=int)

# 4. Extract numbers that are multiples of 3 from a 1D array
multiples_of_3 = array_1d[array_1d % 3 == 0]

# 5. Replace elements in an array: If even, divide by 2; if odd, multiply by 3
transformed_array = np.where(array_1d % 2 == 0, array_1d / 2, array_1d * 3)

# 6. Compute the cumulative sum of a NumPy array
cumulative_sum = np.cumsum(array_1d)

# 7. Normalize an array (min-max scaling)
X = np.random.randint(10, 100, size=10)
X_normalized = (X - X.min()) / (X.max() - X.min())

# 8. Stack two arrays vertically and compute the row-wise sum
a, b = np.random.randint(1, 10, (3, 3)), np.random.randint(1, 10, (3, 3))
vertical_stack = np.vstack((a, b))
row_sums = vertical_stack.sum(axis=1)

# 9. Stack two arrays horizontally and compute column-wise means
horizontal_stack = np.hstack((a, b))
column_means = horizontal_stack.mean(axis=0)

# 10. Create a custom sequence with a repeating pattern [1, 2, 3, 1, 2, 3]
custom_sequence = np.tile([1, 2, 3], 3)

# 11. Get unique values in an array and their counts
a = np.random.randint(1, 5, size=10)
unique_values, counts = np.unique(a, return_counts=True)

# 12. Remove outliers using the interquartile range (IQR)
Q1, Q3 = np.percentile(X, [25, 75])
IQR = Q3 - Q1
filtered_X = X[(X >= Q1 - 1.5 * IQR) & (X <= Q3 + 1.5 * IQR)]

# 13. Find positions where two arrays have the same values
a, b = np.random.randint(1, 5, size=10), np.random.randint(1, 5, size=10)
matching_positions = np.where(a == b)

# 14. Extract values between the 25th and 75th percentile of an array
percentile_values = X[(X >= np.percentile(X, 25)) & (X <= np.percentile(X, 75))]

# 15. Implement a piecewise function using `np.piecewise()`
X = np.arange(-10, 10)
piecewise_result = np.piecewise(X, [X < 0, X >= 0], [lambda x: x**2, lambda x: x + 5])

# 16. Swap the first and last columns of a 2D array
arr = np.random.randint(1, 10, (3, 3))
arr[:, [0, -1]] = arr[:, [-1, 0]]

# 17. Swap the first and last rows of a 2D array
arr[[0, -1], :] = arr[[-1, 0], :]

# 18. Reverse the rows of a 2D array
arr_reversed_rows = arr[::-1]

# 19. Reverse the columns of a 2D array
arr_reversed_columns = arr[:, ::-1]

# 20. Generate 10 random numbers from a normal distribution with mean=50 and std=5
normal_dist = np.random.normal(loc=50, scale=5, size=10)

# 21. Print array with 3 decimal places
np.set_printoptions(precision=3)
arr_three_decimals = np.random.random((5, 3))

# 22. Suppress scientific notation in output
np.set_printoptions(suppress=True, precision=7)
randArr = np.random.random((5, 3)) / 1e3

# 23. Generate 100 values between 0 and 10 and apply a sigmoid function
X = np.linspace(0, 10, 100)
sigmoid = 1 / (1 + np.exp(-X))

# 24. Print full NumPy array without truncation
import sys
np.set_printoptions(threshold=sys.maxsize)  # Use max integer value
arr_full = np.arange(15)
