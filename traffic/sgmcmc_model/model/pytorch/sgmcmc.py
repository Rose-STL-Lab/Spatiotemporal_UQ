"""
Stochastic gradient MCMC implementation based on Diffusion Net
"""

import sys, copy
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, criterion, momentum=0.0, lr=0.1, wdecay=0.0, T=0.05, total=50000, L2_weight = 0.0):
        self.net = net
        self.eta = lr
        self.momentum = momentum
        self.T = T
        self.wdecay = wdecay
        self.V = 0.1
        self.criterion = criterion
        self.total = total
        self.L2_weight = L2_weight

        print("Learning rate: ")
        print(self.eta)
        print("Noise std: ")
        print(self.scale)
        print("L2 penalty:")
        print(self.L2_weight)

    
    def backprop(self, x, y, batches_seen):
        self.net.zero_grad()
        """ convert mean loss to sum losses """
        output = self.net(x, y, batches_seen)
        loss = self.criterion(y, output) * self.total
        loss.backward()
        return loss 
    
    # SGD without momentum
    def step(self, x, y, batches_seen):
        loss = self.backprop(x, y, batches_seen)
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul((self.eta*2.0)**0.5)
            proposal.add_(-0.5*self.scale, param.data)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(self.wdecay, param.data)
            # self.velocity[i].mul_(self.momentum).add_(-self.eta, grads).add_(proposal)
            # param.data.add_(self.velocity[i])
            param.data.add_(-self.eta, grads).add_(proposal)
        return loss.data.item()