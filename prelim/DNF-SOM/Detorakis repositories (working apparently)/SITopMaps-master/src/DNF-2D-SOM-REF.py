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
from matplotlib import rc
from numpy.fft import rfft2, irfft2, ifftshift

rc('text', usetex=True)
rc('font', family='serif')


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
    x = np.linspace(xmin, xmax, n, endpoint=False)
    y = np.linspace(ymin, ymax, n, endpoint=False)
    X, Y = np.meshgrid(x, y)
    X += np.random.uniform(-noise, noise, (n, n))
    X = np.mod(X+1, 1)
    Y += np.random.uniform(-noise, noise, (n, n))
    Y = np.mod(Y+1, 1)
    return X.ravel(), Y.ravel()


def g(x, sigma=1.0):
    """ A simple Gaussian function.
    ** Parameters **

    x : double
        The interval in which the Gaussian will be computed

    sigma : double
        The variance of the Gaussian
    """
    return np.exp(-0.5 * (x/sigma)**2)


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
    np.random.seed(137)

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
    epochs = 35000  # Number of training epochs

    W_min, W_max = 0.00, 1.00     # Weights min/max values for initialization
    Ke = 960.0/(n*n) * 3.72  # Strength of lateral excitatory weights
    sigma_e = 0.1                 # Extent of lateral excitatory weights
    Ki = 960.0/(n*n) * 2.40  # Strength of lateral inhibitory weights
    sigma_i = 1.0                 # Extent of lateral excitatory weights

    # Neural field setup
    # --------------------------------------------
    U = np.random.uniform(0.00, 0.01, (n, n))
    V = np.random.uniform(0.00, 0.01, (n, n))

    W = np.random.uniform(W_min, W_max, (n*n, Rn*Rn))

    # FFT implementation
    # --------------------------------------------
    mean = 0.5
    x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
    X, Y = np.meshgrid(np.linspace(x_inf, x_sup, n+1)[1:],
                       np.linspace(y_inf, y_sup, n+1)[1:])
    Dist = np.sqrt((X-mean)**2 + (Y-mean)**2)
    We = Ke * g(Dist, sigma_e) * alpha
    Wi = Ki * g(Dist, sigma_i) * alpha

    We_fft = rfft2(ifftshift(We[::-1, ::-1]))
    Wi_fft = rfft2(ifftshift(Wi[::-1, ::-1]))

    # Skin Receptors setup
    # --------------------------------------------
    R = np.zeros((Rn*Rn, 2))
    R[:, 0], R[:, 1] = grid(Rn, noise=R_noise)
    np.save('gridxcoord', R[:, 0])
    np.save('gridycoord', R[:, 1])

    plt.plot()
    plt.show()

    # Samples generation
    # --------------------------------------------
    size = epochs
    S = np.random.uniform(0, 1, (size, 2))
    dX = np.abs(R[:, 0].reshape(1, Rn*Rn) - S[:, 0].reshape(size, 1))
    dX = np.minimum(dX, 1-dX)
    dY = np.abs(R[:, 1].reshape(1, Rn*Rn) - S[:, 1].reshape(size, 1))
    dY = np.minimum(dY, 1-dY)
    samples = np.sqrt(dX*dX+dY*dY)/mt.sqrt(2.0)
    samples = g(samples, 0.08)

    # Actual training
    # --------------------------------------------
    # plt.ion()
    for e in range(epochs):
        # Pick a sample
        stimulus = samples[e]

        # Computes field input accordingly
        D = ((np.abs(W - stimulus)).sum(axis=-1))/float(Rn*Rn)
        I = (1.0 - D.reshape(n, n)) * alpha

        # Field simulation until convergence
        for l in range(int(T/dt)):
            V = np.maximum(U, 0.0)
            Z = rfft2(V)
            Le = irfft2(Z * We_fft, (n, n)).real
            Li = irfft2(Z * Wi_fft, (n, n)).real
            U += (-U + (Le - Li) + I) / tau * dt

        # plot_activity(V)

        # Learning
        # --------
        W -= lrate * (Le.ravel() * (W - stimulus).T).T

        if e % 50 == 0:
            print e

        # Field activity reset
        # --------------------
        U = np.random.uniform(0.00, 0.01, (n, n))
        V = np.random.uniform(0.00, 0.01, (n, n))

    np.save('weights', W)

    m = Rn
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, aspect=1)
    R = np.zeros((n*m, n*m))
    for j in xrange(n):
        for i in xrange(n):
            R[j*m:(j+1)*m, i*m:(i+1)*m] = W[j*n+i].reshape(m, m)
    im = plt.imshow(R, interpolation='nearest', cmap=plt.cm.bone_r,
                    vmin=0, vmax=1)
    plt.xticks(np.arange(0, n*m, m), [])
    plt.yticks(np.arange(0, n*m, m), [])
    plt.grid()
    plt.show()
