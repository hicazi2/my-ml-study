import numpy as np

# The cost function measures how well the model (with parameters w and b) fits the training data.
def compute_cost(x ,y ,w ,b ):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost_sum += (f_wb - y[i])**2
    total_cost = cost_sum / (2 * m)

    return total_cost

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

print(f"Example Result: {compute_cost(x_train, y_train, 100, 100)}")