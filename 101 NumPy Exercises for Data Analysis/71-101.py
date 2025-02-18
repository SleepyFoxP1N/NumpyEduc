import numpy as np

# -- 📊 Data Preprocessing & Feature Engineering

# 71. How to standardize (z-score normalize) a NumPy array?
# 👉 Convert a dataset so that its mean is 0 and standard deviation is 1.
x = np.random.randn(100, 5)
x_standardized = (x - x.mean(axis=0) / x.std(axis=0))

# 72. How to normalize a dataset to range [0,1]?
# 👉 Feature scaling before using machine learning models.
x_min, x_max = x.min(axis=0), x.max(axis=0)
x_normalized = (x - x_min) / (x_max - x_min)

# 73. How to perform min-max normalization on each row instead of columns?
# 👉 Useful when data should be row-wise comparable (e.g., NLP embeddings).
x_row_min = x.min(axis=1, keepdims=True)
x_row_max = x.max(axis=1, keepdims=True)
x_row_scaled = (x - x_row_min) / (x_row_max - x_row_min)

# 74. How to create polynomial features in NumPy (withot sklearn)?
# 👉 Generates interaction features manually for regression models.
x_poly = np.hstack([x, x**2, x**3])

# -- 📈 Statistics & Probability

# 75. How to compute the convariance matrix of dataset?
# 👉 Used to analyze relationships between features.
cov_matrix = np.cov(x.T)

# 76. How to compute the correlation matrix of a dataset?
# 👉 Useful for categorical feature analysis.
corr_matrix = np.corrcoef(x.T)

# 77. How to compute the mode of a dataset?
# 👉 Simulates real-world correlated data.
from scipy.stats import mode
mode_value = mode(x, axis=0).mode

# 78. How to generate a synthetic dataset with a specific correlation matrix?
# 👉 Simulates real-world correlated data.
mean = np.zeros(5)
cov_matrix = np.array([[1, 0.8, 0.6, 0.3, 0.1],
                       [0.8, 1, 0.5, 0.2, 0.1],
                       [0.6, 0.5, 1, 0.4, 0.2],
                       [0.3, 0.2, 0.4, 1, 0.6],
                       [0.1, 0.1, 0.2, 0.6, 1]])
x_simulated = np.random.multivariate_normal(mean, cov_matrix, size=100)

# -- 📉 Dimensionality Reduction & Data Transformation

# 79. How to apply Principal Component Analysis (PCA) from scratch?
# 👉 Reduces dimensions while preserving variance.
u, s, vt = np.linalg.svd(x - x.mean(axis=0))
x_pca = x @ vt.T[:, :2]

# 80. How to apply Singular Value Decomposition (SVD) to reduce dimensionality?
# 👉 Useful in recommender systems and image compression.
u, s, vt = np.linalg.svd(x, full_matrices=False)
x_reduced = u[:, :2] @ np.diag(s[:2])

# 81. How to reconstruct data from PCA-reduced features?
x_reconstructed = x_pca @ vt[:2, :] + x.mean(axis=0)

# -- 🤖 Machine Learning Preprocessing

# 82. How to compute TF-IDF manually?
# 👉 Used for text processing in NLP.
termFrequency = x / x.sum(axis=1, keepdims=True)
inverse_documentFrequency = np.log(x.shape[0] / (1 + (x > 0).sum(axis=0)))
TfIdf = termFrequency * inverse_documentFrequency

# 83. How to implement k-means clustering from scratch using NumPy?
# 👉 Common clustering algorithm.
def kmeans(x, k=3, max_iters=100):
    centroid = x[np.random.choice(x.shape[0], k, replace=False)]
    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(x[:, None] - centroid, axis=2), axis=1)
        new_centroids = np.array([x[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids
labels, centroids = kmeans(x, k=3)  

# -- 🛠️ Optimization & Linear Algebra

# 84. How to solve a system of linear equations?
a = np.array([[2, -1], [1, 3]])
b = np.array([0, 9])
solution = np.linalg.solve(a, b)

# 85. How to implement logistic regression manually?
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def logistic_regression(x, y, lr=0.1, epochs=1000):
    weights = np.zeros(x.shape[1])
    for _ in range(epochs):
        y_pred = sigmoid(x @ weights)
        weights -= lr * (x.T @ (y_pred - y)) / len(y)
        return weights
weights = logistic_regression(x, np.random.randint(0, 2, 100))

# -- 🔍 Advanced Challenges (Data Science Jobs)

# 86. How to detect outliers using Z-score method?
z_scores = np.abs((x - x.mean(axis=0)) / x.std(axis=0))
outliers = np.where(z_scores > 3)

# 87. How to perform  bootstrapping (resampling with replacement)?
np.random.seed(42)
bootstrap_sample = np.random.choice(x[:, 0], size=len(x), replace=True)

# 88. How to compute the moving median of a time series?
from scipy.ndimage import median_filter
moving_median = median_filter(x[:, 0], size=3)

# 89. How to commpute PageRank?
# 👉 Used in Google algorithm.
web_pages = 100
p = np.random.rand(web_pages, web_pages)
p /= p.sum(axis=0, keepdims=True) # Normalize transition matrix
initialRank = np.ones(web_pages) / web_pages
for _ in range(100):
    initialRank = 0.85 * (p @ initialRank ) + 0.15 / web_pages

# -- 📊 Advanced Data Processing & Feature Engineering

# 90. How to perfrom robust scaling (median absolute deviation normalization)?
# Unlike standardiation, robust scaling handles outliers better.
median = np.median(x, axis=0)
mad = np.median(np.abs(x - median), axis=0)
x_robust_scaled = (x - median) / mad

# 91. How to generate synthetic categorical data with specified probabilities?
# 👉 Simulate imbalanced categorical data for classification problems.
categories = np.array(['A', 'B', 'C'])
classImbalance_probabilities = np.array([0.7, 0.2, 0.1])
synthetic_labels = np.random.choice(categories, size=1000, p=classImbalance_probabilities)

# -- 📈 Advanced Statistical Analysis

# 92. How to compute the empirical
# 👉 Used for distribution comparisons in statistics.
def empirical_cumulative_distribution_function(data):
    x = np.sort(data)
    y = np.arange(1, len(data + 1) / len(data))
    return x, y
x, y = empirical_cumulative_distribution_function(x[:, 0]) # first column

# 93. How to implement weighted mean and variance?
# 👉 Important for unequal sample importance in weighted datasets.
weights = np.random.rand(x.shape[0])
weighted_mean = np.sum(weights * x[:, 0] / np.sum(weights))
weighted_variance = np.sum(weights * (x[:, 0] - weighted_mean) ** 2 / np.sum(weights))

# 94. How to compute the Mahalanobis distance between points?
from scipy.spatial.distance import mahalanobis
cov_inv = np.linalg.inv(np.cov(x.T))
mahalanobis_dist = [mahalanobis(row, x.mean(axis=0), cov_inv) for row in x]

# -- 📉 Advanced Machine Learning Operations

# 95. How to implement a simple decision boundary function for logistic regression?
# 👉 Used for visualizing classification boundaries
def decision_boundary(weights, x1_range=(-2,2), x2_range=(-2, 2)):
    x1 = np.linspace(*x1_range)
    x2 = (-weights[0] * x1 - weights[2]) / weights[1] 
    return x1, x2

# 96. How to compute class balance metrics (Gini index & entropy)?
# 👉 Measures impurity in decision trees and imbalanced datasets.
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))
class_labels = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
gini_index = gini(class_labels)
entropy_value = entropy(class_labels)

# -- 🤖 Optimization & AI-Related NumPy Applications

# 97. How to optimize a quadratic function using gradient descent?
# 👉 Core idea behind deep learning optimizers.
def gradient_descent(learning_rate=0.1, epochs=100):
    x = np.random.rand() # Start at a random point
    for _ in range(epochs):
        grad = 2 * x # Gradient of f(x) = x^2
        x -= learning_rate * grad
    return x # Should be close to 0 (minimum of x^2)
optimize_x = gradient_descent()

# 98. How to implement a simple Markov Chain in NumPy?
# 👉 Used in text generation, weather forecasting, and reinforcement learning.
states = ['Sunny', 'Rainy', 'Cloudy']
transition_matrix = np.array([[0.8, 0.1, 0.,1],
                              [0.2, 0.6, 0.2],
                              [0.3, 0.3, 0.4]])
state = 0 # Start with 'Sunny'
np.random.seed(42)
for _ in range(10):
    state = np.random.choice(range(3), p=transition_matrix[state])
    print(states[state])
    
# 99. How to apply NumPy broadcasting to batch cosine similarity calculations?
# 👉 Used in recommendations systems & NLP word embeddings.
def cosine_similarity(a, b):  # Lowercase to indicate single vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example usage:
vec1 = np.random.rand(10)
vec2 = np.random.rand(10)
similarity_score = cosine_similarity(vec1, vec2)  # Works for single vectors
def batchCosine_similarity(A, B):
    return np.einsum('ij,ij->i', A, B) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))

# Now the function works for batch cosine similarity
A = np.random.rand(100, 10)
B = np.random.rand(100, 10)
cosine_similarities = cosine_similarity(A, B)  # Function is now used

# 100. How to solve a linear regression problem using NumPy (Normal equation method)?
# 👉 Alternative to gradient descent for small datasets
X_b = np.c_[np.ones((X.Shape[0], 1)), X] # Add bias term
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ np.random.rand(X.shape[0]) # Solve for θ

# -- 💡 Bonus: AI & Reinforcement Learning

# 101. How to simulate an epsilon-greedy strategy for a multi-armed bandit problem?
# 👉 Used for reinforcement learning for dynamic decision-making.
def epsilon_greedy(Q, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q)) # Explore
    return np.argmax(Q) # Exploits
Q_values = np.zeros(10) # 10 possible actions
action = epsilon_greedy(Q_values, epsilon=0.1)

'''
✅ Summary: What You’ve Learned
By practicing problems 1-101, you now have:

Data processing skills (normalization, outliers, scaling)
Statistical methods (correlation, entropy, Gini index)
Optimization techniques (gradient descent, Markov chains)
AI & ML operations (PCA, logistic regression, multi-armed bandits)
Efficient NumPy tricks (broadcasting, advanced indexing)
'''



