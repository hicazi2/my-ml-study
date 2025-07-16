import numpy as np

# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dense layer implementation
def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]            # j-th column of W (weights for neuron j)
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out

# Example input (a[0])
a_in = np.array([-2, 4])  # x

# Weight matrix (W): shape (2, 3)
W = np.array([
    [1, -3, 5],
    [2,  4, -6]
])

# Bias vector (b)
b = np.array([-1, 1, 2])

# Apply single forward pass
a_out = dense(a_in, W, b)
print("Output of dense layer:", a_out)

def sequential(x):
    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)
    f_x = a4
    return f_x

# Dummy example for sequential()
W1 = np.array([[1, -3, 5], [2, 4, -6]])
b1 = np.array([-1, 1, 2])

W2 = np.random.randn(3, 2)  # Layer 2: 3 inputs → 2 outputs
b2 = np.random.randn(2)

W3 = np.random.randn(2, 3)  # Layer 3: 2 inputs → 3 outputs
b3 = np.random.randn(3)

W4 = np.random.randn(3, 1)  # Output layer: 3 inputs → 1 output
b4 = np.random.randn(1)

x = np.array([-2, 4])
output = sequential(x)
print("Final output:", output)

