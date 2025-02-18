import numpy as np
from scipy.stats import zscore

np.set_printoptions(precision=3)

# 25. Load dataset while handling missing values gracefully
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype=object, encoding='utf-8')

# Ensure dataset shape is correct
iris = iris[~np.isnan(iris).any(axis=1)]

names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
iris_3rows = iris[:3]

# 26. Extract species column and count occurrences
species = iris[:, 4].astype(str)
unique_species, species_counts = np.unique(species, return_counts=True)

# 27. Convert numerical columns into a 2D array and standardize it
from scipy.stats import zscore
iris_2d = iris[:, :4].astype(float)
iris_standardized = zscore(iris_2d, axis=0)  # Z-score normalization

# 28. Compute descriptive statistics (mean, median, standard deviation, variance)
stats = {
    'mean': np.mean(iris_2d, axis=0),
    'median': np.median(iris_2d, axis=0),
    'std_dev': np.std(iris_2d, axis=0),
    'variance': np.var(iris_2d, axis=0)
}

# 29. Perform robust scaling using Median Absolute Deviation (MAD)
median = np.median(iris_2d, axis=0)
mad = np.median(np.abs(iris_2d - median), axis=0)
iris_mad_scaled = (iris_2d - median) / mad

# 30. Compute the softmax score across each row (probability distribution)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

softmax_scores = softmax(iris_2d)

# 31. Compute percentile scores for feature scaling
percentiles = np.percentile(iris_2d, [5, 25, 50, 75, 95], axis=0)

# 32. Randomly insert NaN values into the dataset
np.random.seed(777)
nan_mask = np.random.rand(*iris_2d.shape) < 0.05  # 5% missing values
iris_2d_with_nan = iris_2d.copy()
iris_2d_with_nan[nan_mask] = np.nan

# 33. Find positions of missing values and compute the percentage of missing data
missing_positions = np.where(np.isnan(iris_2d_with_nan))
missing_percentage = np.mean(np.isnan(iris_2d_with_nan)) * 100

# 34. Replace missing values with column median
iris_filled = np.where(np.isnan(iris_2d_with_nan), np.nanmedian(iris_2d_with_nan, axis=0), iris_2d_with_nan)

# 35. Identify and remove outliers using Z-score filtering
z_scores = np.abs(zscore(iris_2d, axis=0))
outlier_mask = np.any(z_scores > 3, axis=1)  # Outliers have Z-score > 3
iris_no_outliers = iris_2d[~outlier_mask]

# 36. Compute correlation matrix between features
correlation_matrix = np.corrcoef(iris_2d.T)

# 37. Identify the most correlated pair of features
corr_pairs = np.triu_indices_from(correlation_matrix, k=1)  # Upper triangle indices
most_correlated = np.argmax(np.abs(correlation_matrix[corr_pairs]))  # Highest absolute correlation
feature_pair = (corr_pairs[0][most_correlated], corr_pairs[1][most_correlated])

# 38. Replace missing values using k-nearest neighbors (KNN imputation)
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=3)
iris_knn_imputed = knn_imputer.fit_transform(iris_2d_with_nan)

# 39. Generate polynomial features up to degree 3
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3, include_bias=False)
iris_poly_features = poly.fit_transform(iris_2d)

# 40. Convert continuous values into discrete bins (categorization)
petal_length_bins = np.digitize(iris_2d[:, 2], bins=[0, 3, 5, 10])
bin_labels = {1: 'short', 2: 'medium', 3: 'long'}
petal_length_categories = np.array([bin_labels.get(x, 'unknown') for x in petal_length_bins])
