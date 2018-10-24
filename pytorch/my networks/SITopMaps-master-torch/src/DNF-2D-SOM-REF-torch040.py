#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2010-2014, Georgios Is. Detorakis (gdetor@gmail.com)
#                          Nicolas P. Rougier (nicolas.rougier@inria.fr)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import rc
from numpy.fft import rfft2, irfft2, ifftshift

rc('text', usetex=True)
rc('font', family='serif')
#device = torch.device("cuda:0" if torch.cuda_is_available() else "cpu")
device = torch.device("cuda:0")


def grid(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, noise=0.0):
    """ Create the receptors regular grid. Jitter can be added.
    ** Parameters **

    n : int
        Size of the grid

    xmin, xmax : double
        The infimum and supremum of the x interval

    ymin, ymax : double
        The infimum and supremum of the y interval

    noise : double
        The jitter to add to the grid
    """
    x = torch.from_numpy(np.linspace(xmin, xmax, n, endpoint=False))
    y = torch.from_numpy(np.linspace(ymin, ymax, n, endpoint=False))
    X, Y = torch.from_numpy(np.asarray(np.meshgrid(x, y))).to(device)
    
#    U = torch.FloatTensor(n,n).uniform_(0.00, 0.01).to(device)
#    V = torch.FloatTensor(n, n).uniform_(0.00, 0.01).to(device)
    
    X += torch.DoubleTensor(n,n).uniform_(-noise, noise).to(device)
    # NOTE: torch.remainder is the same as np.mod (as the remainder result shares the same sign as the divisor; with torch.fmod, the result has the same sign as the dividend)
    X = torch.remainder(X+1, 1)
    Y += torch.DoubleTensor(n,n).uniform_(-noise, noise).to(device)
    Y = torch.remainder(Y+1, 1)
    # return X.view(-1), Y.view(-1) # VERY UNSURE about this; lots of people online advising that .view(-1) is not the same as np.ravel
    return X.view(X.numel()), Y.view(Y.numel())


def g(x, sigma=1.0):
    """ A simple Gaussian function.
    ** Parameters **

    x : double
        The interval in which the Gaussian will be computed

    sigma : double
        The variance of the Gaussian
    """
    return torch.exp_(-0.5 * (x/sigma)**2)


def plot_activity(data):
    """ Plots the activity of the neural field.
    ** Parameters **

    data : double
        The 2d array of the neural activity
    """
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.draw()


if __name__ == '__main__':
    torch.manual_seed(137)
    torch.cuda.manual_seed(137)

    # Parameters
    # --------------------------------------------
    Rn = 16         # Receptors count (Rn x Rn)
    R_noise = 0.05  # Receptors placement noise
    n = 32          # Neural field size (n x n)

    T = 10.0        # 90.0 No of Euler's time discretization
    ms = 0.001
    dt = 100.0 * ms
    lrate = 0.03    # 0.005 Learning rate
    alpha = 0.1     # Time constant
    tau = 1.00      # Synapse temporal decay
    epochs = 1000 # Number of training epochs

    W_min, W_max = 0.00, 1.00     # Weights min/max values for initialization
    Ke = 960.0/(n*n) * 3.72  # Strength of lateral excitatory weights
    sigma_e = 0.1                 # Extent of lateral excitatory weights
    Ki = 960.0/(n*n) * 2.40  # Strength of lateral inhibitory weights
    sigma_i = 1.0                 # Extent of lateral excitatory weights
    
    n_zeros = torch.zeros(n,n) 

    # Neural field setup
    # --------------------------------------------
#    U = np.random.uniform(0.00, 0.01, (n, n))
#    V = np.random.uniform(0.00, 0.01, (n, n))
    Utorch = torch.FloatTensor(n,n).uniform_(0.00, 0.01).to(device)
    Vtorch = torch.FloatTensor(n, n).uniform_(0.00, 0.01).to(device)

#    W = np.random.uniform(W_min, W_max, (n*n, Rn*Rn))
    Wtorch = torch.FloatTensor(n*n, Rn*Rn).uniform_(W_min, W_max).to(device)  # random initialisation of weight matrix

    # FFT implementation
    # --------------------------------------------
    mean = 0.5
    x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
    x_lin_torch = torch.linspace(x_inf, x_sup, n+1)[1:]
    y_lin_torch = torch.linspace(y_inf, y_sup, n+1)[1:]
    Xtorch, Ytorch = torch.from_numpy(np.asarray(np.meshgrid(x_lin_torch, y_lin_torch))).to(device)
    Dist = torch.sqrt((Xtorch-mean)**2 + (Ytorch-mean)**2)
    We_torch = Ke * g(Dist, sigma_e) * alpha
    Wi_torch = Ki * g(Dist, sigma_i) * alpha

#    We_fft = rfft2(ifftshift(We[::-1, ::-1])) # rfft2: numpy function to compute 2D FFT of real array
#    Wi_fft = rfft2(ifftshift(Wi[::-1, ::-1]))
    
    
    # NEED TO SHIFT SOMEHOW
    We_fft_torch = torch.rfft(torch.flip(torch.flip(We_torch, [0]), [1]), 2)
    Wi_fft_torch = torch.rfft(torch.flip(torch.flip(Wi_torch, [0]), [1]), 2)

    # Skin Receptors setup
    # --------------------------------------------
    Rtorch = torch.zeros([Rn*Rn, 2])
    Rtorch[:, 0], Rtorch[:, 1] = grid(Rn, noise=R_noise)
    torch.save(Rtorch[:, 0], 'torch_gridxcoord.pt')
    torch.save(Rtorch[:, 1], 'torch_gridycoord.pt')

#    plt.plot()
#    plt.show()
    # Samples generation
    # --------------------------------------------
    size = epochs
    Storch = torch.FloatTensor(size, 2).uniform_(0, 1).to(device)
    dXtorch = torch.FloatTensor.abs_(torch.reshape(Rtorch[:, 0], (1, Rn*Rn)).to(device) - torch.reshape(Storch[:, 0], (size, 1)).to(device))
    dXtorch = torch.min(dXtorch, 1-dXtorch)
    dYtorch = torch.FloatTensor.abs_(torch.reshape(Rtorch[:, 1], (1, Rn*Rn)).to(device) - torch.reshape(Storch[:, 1], (size, 1)).to(device))
    dYtorch = torch.min(dYtorch, 1-dYtorch)
    samples_torch = torch.sqrt_(dXtorch*dXtorch + dYtorch*dYtorch)/mt.sqrt(2.0)
    samples_torch = g(samples_torch, 0.08)

    # Actual training
    # --------------------------------------------
#    plt.ion()
    for e in range(epochs):
        # Pick a sample
        stimulus_torch = samples_torch[e]

        # Computes field input accordingly
#        D = ((np.abs(W - stimulus)).sum(axis=-1))/float(Rn*Rn)
#        I = (1.0 - D.reshape(n, n)) * alpha
        Dtorch = torch.sum((torch.FloatTensor.abs_(Wtorch - stimulus_torch)), 1)/float(Rn*Rn)
        Itorch = (1.0 - torch.reshape(Dtorch, (n,n))) * alpha
        
        # Field simulation until convergence
        for l in range(int(T/dt)):
#            V = np.maximum(U, 0.0)
            Vtorch = torch.max(Utorch, n_zeros.to(device))
#            Z = rfft2(V)
            Ztorch = torch.rfft(Vtorch, 2)
#            Le = irfft2(Z * We_fft, (n, n))
#            Li = irfft2(Z * Wi_fft, (n, n))
            
            LeTorch = torch.irfft(Ztorch * We_fft_torch, 2, signal_sizes=Vtorch.shape)
            LiTorch = torch.irfft(Ztorch * Wi_fft_torch, 2, signal_sizes=Vtorch.shape)
            
            Utorch += (-Utorch + (LeTorch - LiTorch) + Itorch) / tau * dt

        # plot_activity(V)

        # Learning
        # --------
        Wtorch -= lrate * torch.t((LeTorch.view(LeTorch.numel()) * torch.t(Wtorch - stimulus_torch)))

        if e % 50 == 0:
            print(e)

        # Field activity reset
        # --------------------
#        U = np.random.uniform(0.00, 0.01, (n, n))
#        V = np.random.uniform(0.00, 0.01, (n, n))
        Utorch = torch.FloatTensor(n,n).uniform_(0.00, 0.01).to(device)
        Vtorch = torch.FloatTensor(n,n).uniform_(0.00, 0.01).to(device)

    torch.save(Wtorch, 'torch_weights.pt')
#    np.save('weights', Wtorch)

    m = Rn
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, aspect=1)
#    R = np.zeros((n*m, n*m))
    Rtorch = torch.zeros([n*m, n*n])
    for j in range(n):
        for i in range(n):
            Rtorch[j*m:(j+1)*m, i*m:(i+1)*m] = torch.reshape(Wtorch[j*n+i], (m,m))
    im = plt.imshow(Rtorch, interpolation='nearest', cmap=plt.cm.bone_r,
                    vmin=0, vmax=1)
    plt.xticks(np.arange(0, n*m, m), [])
    plt.yticks(np.arange(0, n*m, m), [])
    plt.grid()
    plt.show()