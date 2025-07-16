import numpy as np

#Z-Score Normalization
def zscore_normalization(x):
    """
    computes  X, zscore normalized by column
    
    Args:
      x (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      x_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column(feature)
    mu = np.mean(x, axis=0) # mu will have shape (n,)
    # find the standard deviation of each column(feature)
    sigma = np.std(x, axis=0) # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example and divide by std for that column
    x_norm = (x - mu) / sigma
    return (x_norm, mu, sigma)


# Example training data: 4 features, 3 examples
X_train = np.array([
    [2104, 3, 1, 45],
    [1600, 3, 2, 40],
    [2400, 4, 1, 30]
])
y_train = np.array([399.9, 329.9, 369.0])  # Example target values (in $1000s)

# Example weights and bias (for demonstration)
w_norm = np.array([100, 10, -20, -1])  # Example weights for each feature
b_norm = 200  # Example bias

# Normalize the original features
X_norm, X_mu, X_sigma = zscore_normalization(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# Predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm
print(f"Predictions for training data: {yp}")

# First, normalize our example house
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(f"Normalized features for example house: {x_house_norm}")
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f"Predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")