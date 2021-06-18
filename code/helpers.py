# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import torch


def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def accuracy(y1, y2):
    return np.mean(y1 == y2)

def prediction(A, w):
    return (A @ w > 0) * 2 - 1

def calculate_accuracy(y, A, w):
    """compute the training accuracy on the training set (can be called for test set as well).
    A: the full dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    """
    predicted_y = prediction(A, w)
    return accuracy(predicted_y, y)

def hinge_loss(y, A, w):
    return np.maximum(1 - y * (A @ w), 0)

def calculate_primal_objective(y, A, w, lambda_):
    """compute the full cost (the primal objective), that is loss plus regularizer.
    A: the full dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    
    Note - Function designed for PyTorch
    """
    if type(A) == torch.Tensor:
        v = torch.maximum(1 - y * (A @ w), torch.zeros_like(y))
#         print(v.requires_grad)
        loss = torch.sum(v) + lambda_ / 2 * torch.sum(w ** 2)
#         print(w.requires_grad)
#         print(loss.requires_grad)
#         loss.requires_grad = True 
        return loss
    else:
        v = np.maximum(1 - y * (A @ w), 0)
        return np.sum(v) + lambda_ / 2 * np.sum(w ** 2)
        
def calculate_dual_objective(y, A, w, alpha, lambda_):
    """calculate the objective for the dual problem."""
    return np.sum(alpha)  - lambda_ / 2.0 * np.sum(w ** 2) # w = 1/lambda * A * Y * alpha
    
def SVM_loss(y, A, w, lambda_):
    """compute the full cost (the primal objective), that is loss plus regularizer.
    A: the full dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    
    Note: Function designed to be used with NumPy
    """
    v = np.maximum(1 - y * (A @ w), 0)
    return np.sum(v) + lambda_ / 2 * np.sum(w ** 2)

def compute_stoch_gradient_svm(A, y, lambda_, w_t, num_data_points):
    z = A.dot(w_t) * y
    if z >= 1:
        gradient = lambda_ * w_t
    else:
        gradient = lambda_ * w_t - num_data_points * y * A
    return gradient.reshape(-1)

def compute_gradient_svm(A, y, lambda_, w_t, num_data_points):
    z = A.dot(w_t) * y
    gradient = np.zeros((A.shape[0], w_t.shape[0]))

    gradient[z >= 1, :] = (0.01 * w_t)
    gradient[z < 1, :] = (np.matlib.repmat(0.01 * w_t, A.shape[0], 1)  - (A*(np.matlib.repmat(y, 300, 1).T)) * num_data_points)[z < 1, :]
    return gradient.mean(axis = 0)

def project_to_box(tensor):
    return min(max(tensor, 0.0), 1.0)

def calculate_coordinate_update_wngrad(y, A, lambda_, alpha, w, i, stepsize):      
    # calculate the update of coordinate at index=i.
    a_i, y_i = A[i], y[i]
    old_alpha_i = np.copy(alpha[i])
    
    gradi = 1 - 1/lambda_*(a_i.T @ a_i)*old_alpha_i \
            - 1/lambda_*(lambda_ * y_i * a_i.T @ w - (a_i.T @ a_i)*old_alpha_i);
    b = 1/stepsize
    alpha[i] = project_to_box(old_alpha_i + stepsize*gradi)
    b = b + gradi*gradi / b
    
    # compute the corresponding update on the primal vector w
    w += (1.0 / lambda_) * (alpha[i] - old_alpha_i) * y_i * a_i
    return w, alpha, 1/b