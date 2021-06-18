# Bunch of useful libraries + wngrad
from collections import defaultdict
import numpy as np
import numpy.matlib
import copy
import scipy
import scipy.sparse as sps
import math
import matplotlib.pyplot as plt
import time
import torch
from sklearn.datasets import load_svmlight_file
import random
import helpers

class WNGrad(torch.optim.Optimizer):
  
    def __init__(self, params, b = None, lambda_wn=None, b_sq = False):
#         if b == None:
#             raise ValueException('Please provide a b parameter.')
        self.b = b
        self.b_sq = b_sq
        defaults = dict(lambda_wn=lambda_wn)
        super(WNGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(WNGrad, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure==None:
            raise Exception('WNGrad requires closure to work.')
        
        with torch.enable_grad():
            loss = closure()
            loss.backward()
            
        # Iterate groups of parameters (aka layers in NN) and update
        for group in self.param_groups:
            # Iterate actual parameters in the layers 
            for param in group['params']:
                if param.grad is not None:
                    # Update
                    param.sub_(param.grad, alpha=1/self.b) #add_ is inplace.
        
        # Calculate loss again with new params
        with torch.enable_grad():
            loss = closure()
            loss.backward()
        # Iterate parameters and flatten
        grad_flat = torch.empty((0,))
        for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        grad_flat = torch.cat((grad_flat, param.grad.flatten()))
        # Update b with norm of new grad
        if self.b_sq:
            grad_2_norm = grad_flat.pow(2).sum()
        else:
            grad_2_norm = grad_flat.pow(2).sum().sqrt()
        self.b = self.b + grad_2_norm/self.b

        return loss
    
class WNGrad_C(torch.optim.Optimizer):
  
    def __init__(self, params, b=1, lambda_=None, C=1):

        defaults = dict(b=b, lambda_=lambda_, C=C)
        super(WNGrad_C, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(WNGrad, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:# This enables us to re-evaluate the gradient.
            with torch.enable_grad():
                loss, iteration = closure() 

        withoutLambda = True
        grad_size = None
        b_ret = None
        use_random_init = False

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    state = self.state[param]
                    lambda_ = group['lambda_']
                    C = group['C']
                    if len(state) == 0:
                        if use_random_init:
                          #This is where we sample N points around x_1 to estimate L
                          mean = param.data
                          cov = np.eye(mean.shape[0])
                          random_points = np.random.multivariate_normal(mean, cov, mean.shape[0])
                          print(random_points)
                          state['b_j'] = np.linalg.norm(param.grad, ord=2)*C
                        else:
                          # Otherwise just scale by C
                          print(iteration, group['b'], np.linalg.norm(param.grad, ord=2))
                          state['b_j'] = np.linalg.norm(param.grad, ord=2)*C

                    b = state['b_j']
                    b_ret = b
                    if lambda_ is None:
                        #print("Before", param.grad)
                        param.sub_(param.grad*0.5, alpha=1/b) #add_ is inplace.
                        #print("After sub", param, param.grad)
                    else:
                        # Scaling used (opposite order)
                        withoutLambda = False
                        grad_2_norm = np.linalg.norm(param.grad, ord=2)**2
                        state['b_j'] = b + (lambda_**2)*grad_2_norm/b
                        param.sub_(param.grad, alpha=lambda_/b)
        
        # Re evaluate the gradients so that we can update b_j based on x_t.
        if closure is not None and withoutLambda: 
            with torch.enable_grad():
                _,_ = closure()
                # Update b_j for each param using gradient eval at x_t
                for group in self.param_groups:
                    for param in group['params']:
                        if param.grad is not None:
                            state = self.state[param]
                            prev_bj = state['b_j']
                            grad_2_norm = np.linalg.norm(param.grad, ord=2)**2
                            grad_size = grad_2_norm
                            #print("After", param.grad)
                            update = grad_2_norm/prev_bj
                            #print("ratio", ratio)
                            state['b_j'] = prev_bj + update
                            b_ret = state['b_j']

        return loss, grad_size, b_ret
