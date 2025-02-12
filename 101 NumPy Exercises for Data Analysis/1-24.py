import numpy as np

# 1. Print NumPy version
print(np.__version__)

# 2. Create a 1D array
array_1d = np.arange(10)

# 3. Create a boolean array
bool_elements = np.ones((3, 3), dtype=bool)

# 4. Extract odd elements from 1D array
odd_elements = array_1d[array_1d % 2 == 1]

# 5. Replace odd elements with -1
array_replaced = array_1d.copy()
array_replaced[array_1d % 2 == 1] = -1

# 6. Replace odd elements without modifying the original array
odd_negativeOne = np.where(array_1d % 2 == 1, -1, array_1d)

# 7. Split array into two halves
first_half, second_half = np.split(array_1d, 2)

# 8. Stack two arrays vertically
a, b = np.arange(10).reshape(2, -1), np.full((2, 5), 1)
vertical_stack = np.vstack((a, b))

# 9. Stack two arrays horizontally
horizontal_stack = np.hstack((a, b))

# 10. Generate custom sequences
seq = np.array([1, 2, 3])
custom_sequence = np.concatenate((np.repeat(seq, 3), np.tile(seq, 3)))

# 11. Get common elements between two arrays
a, b = np.array([1,2,3,2,3,4,3,4,5,6]), np.array([7,2,10,2,7,4,9,4,9,8])
common_items = np.intersect1d(a, b)

# 12. Remove elements in `a` that exist in `b`
unique_items = np.setdiff1d(a, b)

# 13. Find positions where elements match
same_elements_positions = np.where(a == b)

# 14. Extract elements within a range
a = np.array([2, 6, 1, 9, 10, 3, 27])
extracted_elements = a[(5 <= a) & (a <= 10)]

# 15. Vectorize a scalar function
maxx = np.vectorize(lambda x, y: max(x, y))
a, b = np.array([5,7,9,8,6,4,5]), np.array([6,3,4,8,9,7,1])
higher_values = maxx(a, b)

# 16. Swap two columns in a 2D array
arr = np.arange(9).reshape(3,3)
arr[:, [0, 1]] = arr[:, [1, 0]]

# 17. Swap two rows in a 2D array
arr[[0, 1], :] = arr[[1, 0], :]

# 18. Reverse the rows of a 2D array
arr_reversed_rows = arr[::-1]

# 19. Reverse the columns of a 2D array
arr_reversed_columns = arr[:, ::-1]

# 20. Create a 2D array of random floats between 5 and 10
random_floats = np.random.uniform(5, 10, size=(5,3))

# 21. Print array with 3 decimal places
np.set_printoptions(precision=3)
arr_three_decimals = np.random.random((5,3))

# 22. Suppress scientific notation in output
np.set_printoptions(suppress=True, precision=7)
randArr = np.random.random((5,3)) / 1e3

# 23. Limit number of printed items
np.set_printoptions(threshold=5)
arr_limited = np.arange(777)

# 24. Print full NumPy array without truncation
import sys
np.set_printoptions(threshold=sys.maxsize)  # Use max integer value
arr_full = np.arange(15)
