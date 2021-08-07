"""
Stochastic gradient NHT
"""

import sys, copy
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, criterion, momentum=0.0, lr=0.1, wdecay=0.0, total=50000, L2_weight = 0.0, prior_weight = 0.02, A = 0.0):
        self.net = net
        self.eta = lr
        self.momentum = momentum
        self.wdecay = wdecay
        self.V = 0.1
        self.criterion = criterion
        self.total = total
        self.L2_weight = L2_weight
        self.xi = []
        self.velocity = []
        self.velocity_has_set = False
        self.xi_has_set = False
        self.A = A
        self.weights_has_initialized = False
        # init constant
        self.phi = 1.0
        self.annealing_rate = 1.
        self.prior_weight = prior_weight

        self.beta = 0.5 * self.V * self.eta
        self.alpha = 1 - self.momentum
        
        if self.beta > self.alpha:
            sys.exit('Momentum is too large')
        
        self.scale = np.sqrt(2.0 * A) * lr
        print("Learning rate: ")
        print(self.eta)
        print("Noise std: ")
        print(self.scale)
        print("L2 penalty:")
        print(self.L2_weight)
        
    def feed_model_data(self, model, grid_data_seq):
        _, hidden_states = model(grid_data_seq)

        return model, hidden_states

    def generate_forecasts(self, model, hidden_states, init_grid_data, seq_len=48):
        prev_grid = init_grid_data
        meo_forecasts = []

        for i in range(seq_len):
            prev_grid, hidden_states = model(prev_grid, hidden_states)
            meo_forecasts.append(prev_grid)

        return torch.cat(meo_forecasts, 1)
        
    def annealPhi(self):
        self.phi /= self.annealing_rate

    # Is this calculating the mean or sum loss? 
    def backprop(self, x_0, x, y):
        self.net.zero_grad()
        """ convert mean loss to sum losses """
        model, hidden_states = self.feed_model_data(self.net, x)
        meo_forecasts = self.generate_forecasts(self.net, hidden_states, x_0)
        loss_meo = self.criterion(meo_forecasts[:,:,:5], y[:,:,:5])/5
        loss_pm = self.criterion(meo_forecasts[:,:,5:], y[:,:,5:])
        loss = (loss_meo+loss_pm) *  self.total
        loss.backward()
        return loss 
    
    def step(self, x_0, x, y):
        loss = self.backprop(x_0, x, y)
        # Special check for velocity term
        if self.velocity_has_set == False:
            for param in self.net.parameters():
                p = torch.zeros_like(param.data)
                self.velocity.append(p)
            self.velocity_has_set = True
        # End of special check
        # Special check for xi term
        if self.xi_has_set == False:
            for param in self.net.parameters():
                self.xi.append(self.A)
            self.xi_has_set = True
        # End of special check
        
        # Random initialization
        if self.weights_has_initialized == False:
            for param in self.net.parameters():
                torch.nn.init.normal_(param, 0.0, 0.08)
            self.weights_has_initialized = True
        # Update parameters
        for i, param in enumerate(self.net.parameters()):
            ###################### proposal random noise #####################
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(self.scale)

            ##################### Gaussian prior #####################
            if self.prior_weight != 0.0:
                proposal.add_(-self.prior_weight*self.total*0.5*self.eta/4.0, param.data)
                
            ##################### End of prior #######################
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(self.wdecay, param.data)
                
            ###################### Update of velocity #####################
            self.velocity[i].add_(-self.xi[i]*self.eta, self.velocity[i]).add_(-self.eta/self.phi, grads).add_(proposal)
            
            ##################### Update of parameter #####################
            param.data.add_(self.eta, self.velocity[i])
            
            ##################### Update of Xi (thermostat) #######################
            # print(param.data.shape.tolist()[1])
            n = param.data.flatten().shape[0] * 1.0
            self.xi[i] = self.xi[i] + (1.0/(n) * np.matmul(self.velocity[i].flatten().cpu().numpy().T, self.velocity[i].flatten().cpu().numpy()) - 1) * self.eta
        # return loss
        return loss.data.item()
