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
