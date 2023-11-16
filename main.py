import pandas as pd
import numpy as np
import csv as csv
import copy
import math
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
#print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 
print(heart_disease.variables.missing_values)

x_input = pd.read_csv('processed.cleveland.data.csv') 

last_column = int(x_input.shape[1]) - 1
y_input = x_input.iloc[:, last_column]
x_input.drop(x_input.columns[[last_column-1]], axis=1, inplace=True)

w_variables = np.ones(int(x_input.shape[1]))
b_variable = 0

def prediction_linear_model (w, x, b):
    p = np.dot(x, w) + b
    return p    

f = prediction_linear_model(w_variables, x_input, b_variable)

def compute_cost (x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    f_wb_i = prediction_linear_model (w, x, b)
    for index in range (m):
        cost += (f_wb_i[index] - y[index])**2
    cost = cost / (2*m)
    return cost

cost = compute_cost(x_input, y_input, w_variables, b_variable)
print (f'Cost at optimal w : {cost}')

def compute_gradient(x, y, w, b): 

    m,n = x.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):           
        err = (prediction_linear_model(w, x.iloc[i, :], b)) - y[i]
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * x.iloc[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

tmp_dj_db, tmp_dj_dw = compute_gradient(x_input, y_input, w_variables, b_variable)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient detmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')scent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

# initialize parameters
initial_w = np.zeros_like(w_variables)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(x_input, y_input, w_variables, b_variable, compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_input.shape

for i in range(m):
    print(f"prediction: {np.dot(x_input.iloc[i, :], w_final) + b_final:0.2f}, target value: {y_input.iloc[i]}")

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()