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
""" Model response. Calculates the response of the DNF-2D-SOM-REF """
import math as mt
import numpy as np
from sys import stdout
from numpy.fft import rfft2, irfft2, ifftshift


def progress_bar(width, percent):
    """ Display the progress bar.

    ** Parameters **
    width : int
        The width of the progress bar

    percent : int
        The current time step
    """
    marks = mt.floor(width * (percent/100.0))
    spaces = mt.floor(width - marks)
    loader = '[' + ('#' * int(marks)) + (' ' * int(spaces)) + ']'

    stdout.write("%s %d%%\r" % (loader, percent))
    if percent >= 100:
        stdout.write("\n")
    stdout.flush()


def g(x, sigma=1.0):
    """ A Gaussian function.

    ** Parameters **

    x : double
        The intervan in which the function computes the Gaussian

    sigma : double
        The variance of the Gaussian
    """
    return np.exp(-0.5*(x/sigma)**2)


def area_of_activity(data):
    """ It returns the number of active neurons.

    ** Parameters **

    data : double
        The 2d or 1d array of the neural activity
    """
    return sum(1 for i in data.flatten() if i > 0.0)


def dnfsom_activity(n, Rn, l, tau, T, alpha):
    """ It returns the activity of the neural field over a set of stimuli.

    ** Parameters **

    n : int
        Size of neural field

    Rn : int
        Size of receptors grid

    l : int
        Size of receptive fields

    tau : double
        Decay synapse constant

    T : double
        Total simulation time (s)

    alpha : double
        Liptschitz parameter
    """
    ms = 0.001  # ms definition
    dt = 100.0 * ms     # Euler's time step

    # Files to be loaded
    filename = 'weights.npy'
    filenames = 'model_response_64'

    # Allocation of arrays and loading necessary files
    O = np.zeros((l*n, l*n))
    W = np.load(filename)
    V = np.random.random((n, n)) * .01
    U = np.random.random((n, n)) * .01
    Rx = np.load('gridxcoord.npy')
    Ry = np.load('gridycoord.npy')

    # FFT implementation
    mean = 0.5
    Ke, Ki = 3.73, 2.40
    sigma_e, sigma_i = 0.1, 1.0
    x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
    X, Y = np.meshgrid(np.linspace(x_inf, x_sup, n+1, endpoint=True)[1:],
                       np.linspace(y_inf, y_sup, n+1, endpoint=True)[1:])
    D = np.sqrt((X-mean)**2 + (Y-mean)**2)
    We = Ke * g(D, sigma_e) * alpha * 960.0/(n*n)
    Wi = Ki * g(D, sigma_i) * alpha * 960.0/(n*n)
    We_fft = rfft2(ifftshift(We[::-1, ::-1]))
    Wi_fft = rfft2(ifftshift(Wi[::-1, ::-1]))

    # Stimuli generation
    S = np.zeros((l*l, 2))
    for i, x in enumerate(np.linspace(0.0, 1.0, l)):
        for j, y in enumerate(np.linspace(0.0, 1.0, l)):
            S[i*l+j, 0] = x
            S[i*l+j, 1] = y
        dX = np.abs(Rx.reshape(1, Rn*Rn) - S[:, 0].reshape(l*l, 1))
        dX = np.minimum(dX, 1-dX)
        dY = np.abs(Ry.reshape(1, Rn*Rn) - S[:, 1].reshape(l*l, 1))
        dY = np.minimum(dY, 1-dY)
        samples = np.sqrt(dX*dX+dY*dY)/mt.sqrt(2.0)
        samples = g(samples, 0.08)

    # Calculation of model response
    step = 0
    jj = 100.0/(float(l))
    for i in range(l):
        for j in range(l):
            D = ((np.abs(W - samples[i*l+j])).sum(axis=-1))/float(Rn*Rn)
            I = (1.0 - D.reshape(n, n)) * alpha
            for k in range(int(T/dt)):
                V = np.maximum(U, 0.0)
                Z = rfft2(V)
                Le = irfft2(Z * We_fft, (n, n)).real
                Li = irfft2(Z * Wi_fft, (n, n)).real
                U += (-U + (Le - Li) + I) / tau * dt
            O[i*n:(i+1)*n, j*n:(j+1)*n] = np.maximum(U, 0.0)
            V = np.random.random((n, n)) * .01
            U = np.random.random((n, n)) * .01
        step += jj
        progress_bar(30, step)

    np.save(filenames, O)

if __name__ == '__main__':
    np.random.seed(137)
    model_size = 32
    rf_resolution = 64
    num_receptors = 16
    T = 10.0
    tau = 1.0
    alpha = 0.1

    dnfsom_activity(model_size, num_receptors, rf_resolution, tau, T, alpha)
