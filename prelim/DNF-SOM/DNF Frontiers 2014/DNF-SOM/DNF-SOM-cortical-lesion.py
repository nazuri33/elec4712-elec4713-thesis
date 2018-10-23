#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#               Georgios Is. Detorakis (Georgios.Detorakis@inria.fr)
#
# Self-organizing dynamic neural field.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL: http://www.cecill.info.
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
#     python > 2.6 (required): http://www.python.org
#     numpy        (required): http://numpy.scipy.org
#     matplotlib   (required): http://matplotlib.sourceforge.net
#
# -----------------------------------------------------------------------------
# Contributors:
#
#     Nicolas P. Rougier
#     Georgios Is. Detorakis
#
# Contact Information:
#
#     Nicolas P. Rougier
#     INRIA Nancy - Grand Est research center
#     CS 20101
#     54603 Villers les Nancy Cedex France
#
# References:
#
# -----------------------------------------------------------------------------
''' Simulation of a self-organizing neural field.

This script implements the numerical integration of a dynamic neural field
of the form:
                  
  ∂U(x,t)               ⌠+∞                        
τ ------- = -U(x,t) + α ⎮  Wl(|x-y|).f(U(y,t)).dy + α I(x,t)
    ∂t                  ⌡-∞                        


where U(x,t) is the potential of a neural population at position x and time t
      Wl(d) = We(d) - Wi(d) is a neighborhood function from ℝ⁺ → ℝ
      f(u) is the firing rate of a single neuron from ℝ → ℝ
      I(x,t) is the input at position x and time t
      τ is the temporal decay of the synapse
      α is a scaling term

For any stimulus s(t), we have:

I(x,t) = (1 - |s(t)-Wf(x,t)|/n) * G(x)

where n is the size of the stimulus, Wf(x,t) are feedforward weights and G is a
corrective Gaussian function.


 ∂Wf(x,t)                                               ⌠+∞                       
 ------- =  γ Le(x,t)(s(x) - Wf(x))  with: Le(x,t) =  γ ⎮  We(|x-y|).f(U(y,t)).dy 
    ∂t                                                  ⌡-∞                        

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.fftpack import fft2, ifft2, ifftshift

rc('text', usetex=True)
rc('font', family='serif')

def stimulus(shape, width=(0.1,0.1), center=(0.0,0.0)):
    if type(shape) in [float,int]:
        shape = (shape,)
    if type(width) in [float,int]:
        width = (width,)*len(shape)
    if type(center) in [float,int]:
        center = (center,)*len(shape)
    grid=[]
    for size in shape:
        grid.append (slice(0,size))
    C = np.mgrid[tuple(grid)]
    R = np.zeros(shape)
    for i,size in enumerate(shape):
        if shape[i] > 1:
            R += (((C[i]/float(size-1))*2 - 1 - center[i])/width[i])**2
    return 5 * np.exp(-R/2)

def regular(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, noise=0.0):
    _X = (np.resize(np.linspace(xmin,xmax,n),(n,n))).ravel()
    _Y = (np.resize(np.linspace(ymin,ymax,n),(n,n)).T).ravel()
    X = _X + np.random.uniform(-noise, noise, n*n)
    Y = _Y + np.random.uniform(-noise, noise, n*n)
    Imin, Imax = np.argwhere(X < xmin), np.argwhere(X > xmax)
    while len(Imin) or len(Imax):
        X[Imin] = _X[Imin] + np.random.uniform(-noise, noise, len(Imin))
        X[Imax] = _X[Imax] + np.random.uniform(-noise, noise, len(Imax))
        Imin, Imax = np.argwhere(X < xmin), np.argwhere(X > xmax)
    Imin, Imax = np.argwhere(Y < ymin), np.argwhere(Y > ymax)
    while len(Imin) or len(Imax):
        Y[Imin] = _Y[Imin] + np.random.uniform(-noise, noise, len(Imin))
        Y[Imax] = _Y[Imax] + np.random.uniform(-noise, noise, len(Imax))
        Imin, Imax = np.argwhere(Y < ymin), np.argwhere(Y > ymax)
    return X.reshape(n,n),Y.reshape(n,n)

def gaussian(shape, width=(1,1), center=(0,0)):
    grid=[]
    for size in shape:
        grid.append (slice(0,size))
    C = np.mgrid[tuple(grid)]
    R = np.zeros(shape)
    for i,size in enumerate(shape):
        if shape[i] > 1:
            R += (((C[i]/float(size-1))*2 - 1 - center[i])/width[i])**2
    return np.exp(-R/2)

# Cortical lesion function. This function calculates the extent of 
# the cortical lesion and returns a lesion mask of that size for the 
# weights matrix. It can handle three cortical lesion types (I, II, 
# III) defining the flag 1->Band-shaped lesion at the border of the 
# cortical sheet 
# otherwise->Band-shaped or localized lesion at the mainland of the
# cortical sheet depending on the initial position (begin,begin). 
def cortical_lesion( n, m, percent, flag, begin ):
	lesion 	       = np.ones((n,n,m,m,))
	number_neurons = n * n
	killed_neurons = float(number_neurons) * percent/100.0
	length         = np.sqrt( killed_neurons )
	if killed_neurons > number_neurons:
		print 'You are trying to kill more neurons than you have!'
		return -1
	else:
		if flag==1:
			extent = killed_neurons/n
			lesion[:,begin:begin+int(extent),...] = 0.0
			return lesion.reshape( n * n, m * m )
		else:
			if begin > n - length and begin < 0:
				print 'Wrong beggining!'
				return -1
			else:
				length = int( length )
				lesion[begin:begin+length,begin:begin+length,...] = 0.0
				return ( lesion ).reshape( n * n, m * m )

# Silence of activity. This function it works exactly like the cortical_lesion
# function. 
def cortical_silence( n, percent, flag, begin ):
	silence        = np.ones((n,n))
	number_neurons = n * n
	killed_neurons = float(number_neurons) * percent/100.0
	length         = np.sqrt( killed_neurons )
	if killed_neurons > number_neurons:
		print 'You are trying to kill more neurons than you have!'
		return -1
	else:
		if flag==1:
			extent = killed_neurons/n
			silence[:,begin:begin+int(extent)] = 0.0
			return silence
		else:
			if begin > n - length and begin < 0:
				print 'Wrong beggining!'
				return -1
			else:
				length = int( length )
				silence[begin:begin+length,begin:begin+length] = 0.0
				return silence

if __name__ == '__main__':

    np.random.seed(137)

    # Parameters
    # --------------------------------------------
    Sn      = 64     # Skin spatial discretization (Sn x Sn)
    Rn      = 16     # Receptors count (Rn x Rn)
    R_noise = 0.05    # Receptors placement noise
    n       = 32     # Neural field size (n x n)
    p       = 2*n+1
    s       = 1000   # Number of stimuli samples

    S_width        = +0.15         # Stimulus width
    S_xmin, S_xmax = -0.75, +0.75  # Stimulus position xmin/xmax
    S_ymin, S_ymax = -0.75, +0.75  # Stimulus position ymin/ymax

    T	    = 100    # No of Euler's time discretization
    dt      = 20.0/float(T)   # Timestep
    lrate   = 0.05 # Learning rate
    alpha   = 0.100  # Time constant
    tau     = 1.000  # Synapse temporal decay
    epochs  = 10000  # Number of training epochs

    Ke      = 960.0/(n*n) * 3.65  # Strength of lateral excitatory weights
    sigma_e = 0.1                 # Extent of lateral excitatory weights
    Ki      = 960.0/(n*n) * 2.40  # Strength of lateral inhibitory weights
    sigma_i = 1.0                 # Extent of lateral excitatory weights
    Kc      = 1.0                 # Strength of field attenuation
    sigma_c = 2.1                 # Extent of field attenuation

    # Building lesion masks
    lesion  = cortical_lesion( n, Rn, 25, 1, 0 )
    silence = cortical_silence( n, 25, 1, 0 ) 

    # Neural field setup
    # --------------------------------------------
    Ke *= alpha 
    Ki *= alpha
    Ke_fft = fft2(ifftshift(Ke * gaussian((p,p),(sigma_e,sigma_e))))
    Ki_fft = fft2(ifftshift(Ki * gaussian((p,p),(sigma_i,sigma_i))))
    U = np.random.uniform(0.00,0.01,(n,n))
    V = np.random.uniform(0.00,0.01,(n,n))
    U *= silence
    V *= silence
    G = Kc*gaussian((n,n),(sigma_c,sigma_c))
    W = np.genfromtxt( 'weights10000.dat' )
    W *= lesion

    # Skin setup
    # --------------------------------------------
    S = np.zeros((Sn,Sn), dtype=np.double)

    # Receptors setup
    # --------------------------------------------
    R = np.zeros((Rn,Rn), dtype=[('x',np.double),('y',np.double)])
    R['x'] = np.genfromtxt( 'gridxcoord.dat' )
    R['y'] = np.genfromtxt( 'gridycoord.dat' )
    
    # Samples generation
    # --------------------------------------------
    Sx = np.round(R['x']*(Sn-1)).astype(int)
    Sy = np.round(R['y']*(Sn-1)).astype(int)
    samples = np.zeros((Rn*Rn,Rn*Rn))
    ii = 0
    for i,x in enumerate(np.linspace(-0.75,+0.75,Rn)):
	    for j,y in enumerate(np.linspace(-0.75,+0.75,Rn)):
		    S = stimulus((Sn,Sn), width=(0.15,0.15), center = (x,y) )
		    samples[ii] = S[Sx,Sy].ravel()
		    ii += 1

    m = Rn
    plt.ion()
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, aspect=1)
    R = np.zeros((n*m,n*m))
    for j in range(n):
        for i in range(n):
            R[j*m:(j+1)*m,i*m:(i+1)*m] = W[j*n+i].reshape(m,m)
    im = plt.imshow(R, interpolation='nearest', cmap=plt.cm.bone_r,vmin=0,vmax=1)
    plt.xticks(np.arange(0,n*m,m),[])
    plt.yticks(np.arange(0,n*m,m),[])

    plt.text(0.5, 1.1, "Self organizing dynamic neural fields",
             fontsize=16,
             horizontalalignment="center",
             verticalalignment="top",
             transform = ax.transAxes)
    plt.text(0.5, 1.065, "Georgios Is. Detorakis and Nicolas P. Rougier",
             fontsize=12,
             horizontalalignment="center",
             verticalalignment="top",
             transform = ax.transAxes)

    plt.text(0.5, -0.05, "Cortical lesion case",
             fontsize=16,
             horizontalalignment="center",
             verticalalignment="top",
             transform = ax.transAxes)

    plt.text(0.5, -0.11, r'Parameters: Ke = 3.42, Ki = 2.25, $\sigma_e$ = 0.1, $\sigma_i$ = 1.0, $\mu_c$ = 0.0, $\sigma_c$ = 2.1, $\sigma$ = 0.15, dt = 0.2, $\alpha$ = 0.1, $\tau$ = 1.0, $\gamma$ = 0.05',
             fontsize=12,
             horizontalalignment="center",
             verticalalignment="top",
             transform = ax.transAxes)

    plt.grid()
    plt.draw()


    # Actual training
    # --------------------------------------------
    for e in range(epochs):
        # Pick a random sample
        stimulus = samples[np.random.randint(0,ii)]

        # Computes field input accordingly
        D = ((np.abs(W-stimulus)).sum(axis=-1))/float(Rn*Rn)
        I = (1.0-D.reshape(n,n))*G * alpha

        # Field simulation until convergence
        Z = np.zeros((p,p))
        for l in range( T ):
            Z[n//2:n//2+n,n//2:n//2+n] = V
            Zf = fft2(Z)
            Li = (ifft2(Zf*Ki_fft).real)[n//2:n//2+n,n//2:n//2+n]
	    Li *= silence
            Le = (ifft2(Zf*Ke_fft).real)[n//2:n//2+n,n//2:n//2+n]
	    Le *= silence
            U += tau * dt * (-U + I+Le-Li)
	    U *= silence
            V = np.maximum(U, 0)
	    V *= silence

        # Learning
        # --------
        W -= lrate * (Le.ravel() * (W-stimulus).T).T
	W *= lesion

	# Saving weights
	if e%1000==0:
		np.savetxt( 'corles_weights'+str( '%05d' % e )+'.dat', W )
        # Field activity reset
        # --------------------
        U  = np.random.uniform(0.00,0.01,(n,n))
        V  = np.random.uniform(0.00,0.01,(n,n))
	U *= silence
	V *= silence

        # Figure update
        # -------------
        for j in range(n):
            for i in range(n):
                R[j*m:(j+1)*m,i*m:(i+1)*m] = W[j*n+i].reshape(m,m)
        im.set_data(R)
        plt.draw()

plt.ioff()
plt.show()
