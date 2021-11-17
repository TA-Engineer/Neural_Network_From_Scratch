''''
This is the script is designed to build a logistic regression model 
with Neural netowrk utilizing only python and Numpy library. 

Now high level ML framework such as tensorflow or anythin will be used.
'''

# Import required libraries.

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_x, train_y, test_x, test_y, classes = load_dataset()



def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def relu(z):
    s = max(0,z)
    return s

def initialize_with_zero(dim):
    # here dim is the size the weight vector
    w = np.zeros((dim,1)) 
    # creating a x by 1 vector as we want 1 long vector.
    # Initialize bias as zero
    b = 0

    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w, b

dim = 2
w, b = initialize_with_zero(dim)


# looking at how many training examples we have:

num_train = train_x.shape[0]
num_test = train_x.shape[0]
image_H_W = train_x.shape[1]

print(train_x.shape)
print(test_x.shape[0])

# Reshaping image: we will reshape (255,255,3) to (255*255*3,1)

train_set_x_flatten = train_x.reshape(train_x.shape[0],-1).T
test_set_x_flatten = test_x.reshape(test_x.shape[0],-1).T

print(test_x.shape)

# normalizing the iamges:

train_x = train_set_x_flatten/255.
test_x = test_set_x_flatten/255.


# doing a forward propogation

def propagate(w, b, X, Y):

    m = X.shape[1] # as we want to know how many there are for division

    # Forward Propagation

    A = sigmoid((np.dot(w.T,X))+b)
    
    cost = (-1/m)*np.sum( Y*np.log(A)+((1-Y)*np.log(1-A)) )

    dw = (1/m)*np.dot(X, (A-Y).T)
    db = np.sum((1/m)*(A-Y))

    assert(dw.shape == w.shape)
    assert(db.shape == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw":dw,
             "db":db}
    return grads, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

grads, cost = propagate(w, b, X, Y)

##############################################
############### OPTIMIZATION #################
##############################################

def optimize(w, b, X, Y, num_iteration, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iteration):

        grad, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)

        if i%100 == 0:
            costs.append(cost)
        
        if print_cost and I%100 == 0:
            print("Cost after iteration %i: %f" %(i,cost))

    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# optimizing the value
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T, X)+b)

    for i in range(A.shape[1]):

        if (A[0.i]<=0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
        pass

    return Y_prediction
