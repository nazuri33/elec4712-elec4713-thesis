#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#               Georgios Detorakis (Georgios Detorakis@inria.fr)
#
# Self-Organizing Dynamic Neural Field
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
#
# Dependencies:
#
#     python     >= 2.7 (required): http://www.python.org
#     numpy      >= 1.5 (required): http://numpy.scipy.org
#     scipy      >= 0.8 (required): http://www.scipy.org
#     matplotlib >= 1.0 (optional): http://matplotlib.sourceforge.net
#
# -----------------------------------------------------------------------------
# Contributors:
#
#     Nicolas P. Rougier
#     Georgios Detorakis
#
# Contact Information:
#
#     Nicolas P. Rougier
#     INRIA Nancy - Grand Est research center
#     CS 20101
#     54603 Villers les Nancy Cedex France
#
# -----------------------------------------------------------------------------
'''
Self-organized dynamic neural fields with infinite propagation speed

This script implements the numerical integration of a dynamic neural field
that self-organizes itself.

  âˆ‚V(x,t)                  âŒ 
Î± ------- = - V(x,t) + Ï„ ( âŽ® [Ke(|x-y|)-Ki(|x-y|)]V(y)dÂ²y + I(x,t) )
    âˆ‚t                     âŒ¡Î©

where # V(x,t) is the potential of a neural population at position x and time t
      # Î© is the domain of integration
      # Ke(x) is the lateral excitation
      # Ki(x) is the lateral inhibition
      # S(x) is the firing rate of a single neuron
      # Î± is the temporal decay of the synapse
      # Ï„ is a free scaling factor
      # I(x,t) is the input at position x and time t

Numerical parameters:
      # n  : space discretisation
      # dt : temporal discretisation (s)

The integration is made over the finite 2d domain Î© discretized into n x n elements.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def gaussian(shape=(25,25), width=(1,1), center=(0,0)):
    ''' Kind of Gaussian '''
    grid=[]
    for size in shape:
        grid.append (slice(0,size))
    C = np.mgrid[tuple(grid)]
    R = np.zeros(shape)
    for i,size in enumerate(shape):
        if shape[i] > 1:
            R += (((C[i]/float(size-1))*2 - 1 - center[i])/width[i])**2
    return np.exp(-R/2)


if __name__ == '__main__':

    # Simulation parameters
    n        = 64     # Spatial discretization
    dt       = 0.05   # Temporal discretization
    epsilon  = 0.05   # Convergence criterion
    lrate    = 0.01   # Learning rate
    epochs   = 20   # Number of samples to process

    # Sample sets

    # Sâˆž (uniform values in [0,1])
    Soo = np.random.uniform(0,1,1000)

    # S2 (2 discrete values)
    S2 = np.random.randint(0,2,1000)/1.0

    # S3 (3 discrete values)
    S3 = np.random.randint(0,3,1000)/2.0

    # S5 (5 discrete values)
    S5 = np.random.randint(0,5,1000)/4.0

    # S10 (10 discrete values)
    S10 = np.random.randint(0,10,1000)/9.0

    # Choosing a sample set
    samples = S10
    

    # Build model
    # Lateral excitatory kernel
    Ke   = 960.0/(n*n) * 1.50 * gaussian( (2*n+1,2*n+1), (0.1,0.1) ) 
    Ke_f = fft2(ifftshift(Ke))

    # Lateral inhibitory kernel
    Ki   = 960.0/(n*n) * 0.75 * gaussian( (2*n+1,2*n+1), (1.0,1.0) )
    Ki_f = fft2(ifftshift(Ki))

    # Afferent connections
    W = np.random.random((n,n))

    # Potentials arrays
    Z = np.zeros((2*n+1, 2*n+1))
    U = np.zeros((n,n))
    V = np.zeros((n,n))

    # Make a new (interactive) figure
    plt.ion()   
    plt.figure(figsize=(12,9))
    Wi = plt.imshow(W, interpolation='bicubic',
                    cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('Self-Organizing Neural Field', fontsize=20)
    plt.draw()

    for i in range(epochs):
        # Get random sample
        sample = samples[np.random.randint(samples.size)]

        # Reset of activity
        dV = 1
        U[...] = np.random.random((n,n))*0.01
        V[...] = np.random.random((n,n))*0.01

        # Set input
    	I = np.abs( sample - W )
    	
    	# Solving DNF equation using Euler Method
        while dV > epsilon:
            Z[n:2*n,n:2*n] = V
            Z_f = fft2(Z)
            Le = ifft2(Z_f*Ke_f).real[n:2*n,n:2*n]
            Li = ifft2(Z_f*Ki_f).real[n:2*n,n:2*n]
            U += (-V + 0.1*(Le-Li+1-I))*dt
            V[...] = np.maximum( U, 0 )
            dV = np.abs(V - Z[n:2*n,n:2*n]).sum()
        # Weights learning
        W += ( lrate * Le * ( sample - W ) )

        # Update figure
        if i%10 == 0:
            Wi.set_data(W)
            plt.draw()
            print 'Epoch %d/%d' % (i,epochs)

plt.ioff()
plt.contour( W, 10, colors='w', linewidths=2 )
plt.show()