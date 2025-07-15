import numpy as np

x_train= np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
# x_train is the input variable (size in 1000 square feet)
print(f"x_train: {x_train}") 
# y_train is the target (price in 1000s of dollars)
print(f"y_train: {y_train}") 

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m=x_train.shape[0]
print(f"Number of the training examples is: {m}")

# It can be also expressed like this:
m = len(x_train)
print(f"Number of the training examples is: {m}")
i= 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

def compute_model_output(x, w ,b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

print(f"Example Result: {compute_model_output(
    x_train, 200, 100)}")

