import numpy as np

# 1. Import numpy as np and see the version
print(np.__version__)

# 2. How to create a 1D array?
array_1d = np.array([0,1,2,3,4,5,6,7,8,9])
array_1d = np.arange(10)

# 3. How to create a boolean array?
bool_elements = np.full((3,3), True, dtype=bool)
bool_elements = np.ones((3,3), dtype=bool)
bool_elements = np.zeros((3,3), dtype=bool)

# 4. How to extract items that satisfy a given condition from 1D array?
odd_elements = array_1d[array_1d % 2 == 1]

# 5. How to replace items that satisfy a condition with another value in numpy array?
odd_negativeOne = array_1d.copy()
odd_negativeOne[array_1d % 2 == 1] = -1

# 6. How to replace items that satisfy a condition without affecting the original array?
odd_negativeOne = np.where(array_1d % 2 == 1, -1, array_1d)

# 7. How to reshape an array?
firstHalf_elements = array_1d[:5]
secondHalf_elements = array_1d[5:]

# 8. How to stack two arrays vertically?
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)
vertical_stack = np.vstack((a, b))

# 9. How to stack two arrays horizontally?
horizontal_stack = np.hstack((a, b))

# 10. How to generate custom sequences in numpy without hardcoding?
a = np.array([1,2,3])
custom_sequence = np.r_[np.repeat(a, 3), np.tile(a, 3)]

# 11. How to get the common items between two python numpy arrays?
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
common_items = np.intersect1d(a, b)

# 12. How to remove from one array those items that exist in another?
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
unique_items = np.setdiff1d(a, b)

# 13. How to get the positions where elements of two arrays match?
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
sameElements_position = np.where(a == b)

# 14. How to extract all numbers between a given range from a numpy array?
a = np.array([2, 6, 1, 9, 10, 3, 27])
extracted_elements = a[(a >= 5) & (a <= 10)]
extracted_elements = a[np.where((a >= 5) & (a <= 10))]
extracted_elements = a[np.where(np.logical_and(a >= 5, a <= 10))]

# 15. How to make a python function that handles scalars to work on numpy arrays?
def maxx(x, y):
    if x > y:
        return x
    else:
        return y
vectorized_maxx = np.vectorize(maxx, otypes=[float])
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
higherValues = vectorized_maxx(a, b)
    
# 16. How to swap two columns in a 2d numpy array?
arr = np.arange(9).reshape(3,3)
arr = arr[:,[1,0,2]]

# 17. How to swap two rows in a 2d numpy array?
arr = np.arange(9).reshape(3,3)
arr = arr[[1,0,2],:]

# 18. How to reverse the rows of a 2D array?
arr = np.arange(9).reshape(3,3)
arr = arr[::-1]

# 19. How to reverse the columns of a 2D array?
arr = np.arange(9).reshape(3,3)
arr = arr[:, ::-1]

# 20. How to create a 2D array containing random floats between 5 and 10?
random_floats = np.random.randint(5, 10, size=(5,3)) + np.random.random((5,3))
random_floats = np.random.uniform(5, 10, size=(5,3))

# 21. How to print only 3 decimal places in python numpy array?
randArr = np.random.random((5,3))
np.set_printoptions(precision=3) 
arr_threeDecimanPlaces = randArr[:4]

# 22. How to pretty print a numpy array by suppressing the scientific notation (like 1e10)?
np.random.seed(777)
randArr = np.random.random([5,3])/1e3
np.set_printoptions(suppress=True, precision=7)

# 23. How to limit the number of items printed in output of numpy array?
#np.set_printoptions(threshold=5)
arr = np.arange(777)

# 24. How to print the full numpy array without truncating?
np.set_printoptions(threshold=None)
arr = np.arange(15)

