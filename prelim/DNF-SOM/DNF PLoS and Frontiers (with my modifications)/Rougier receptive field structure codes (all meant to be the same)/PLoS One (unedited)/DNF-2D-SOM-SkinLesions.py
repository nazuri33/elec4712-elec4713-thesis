#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
#  Contributors: Georgios Is. Detorakis (Georgios.Detorakis@inria.fr)
# 		 Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# Self-organizing dynamic neural field applying skin receptors lesion.
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
#     Georgios Is. Detorakis
#     Nicolas P. Rougier
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from numpy.fft import fft2, ifft2, rfft2, irfft2, fftshift, ifftshift

rc('text', usetex=True)
rc('font', family='serif')

# FFT optimization function.
def best_fft_shape(shape):
	# fftpack (not sure of the base)
	base = [13,11,7,5,3,2]
    	# fftw
    	# base = [13,11,7,5,3,2]

    	def factorize(n):
        	if n == 0:
        	    raise(RuntimeError, "Length n must be positive integer")
        	elif n == 1:
        	    return [1,]
        	factors = []
        	for b in base:
        	    while n % b == 0:
        	        n /= b
        	        factors.append(b)
        	if n == 1:
        	    return factors
        	return []

    	def is_optimal(n):
		factors = factorize(n)
    	    	# fftpack
    	    	return len(factors) > 0
    	    	# fftw
    	    	# return len(factors) > 0 and factors[:2] not in [[13,13],[13,11],[11,11]]

    	shape = np.atleast_1d(np.array(shape))
    	for i in range(shape.size):
		while not is_optimal(shape[i]):
			shape[i] += 1
	return shape.astype(int)

# FFT optimization function.
def extract(Z, shape, position, fill=0):
	R = np.ones(shape, dtype=Z.dtype)*fill
	P  = np.array(list(position)).astype(int)
    	Rs = np.array(list(R.shape)).astype(int)
    	Zs = np.array(list(Z.shape)).astype(int)

    	R_start = np.zeros((len(shape),)).astype(int)
    	R_stop  = np.array(list(shape)).astype(int)
    	Z_start = (P-Rs//2)
    	Z_stop  = (P+Rs//2)+Rs%2

    	R_start = (R_start - np.minimum(Z_start,0)).tolist()
    	Z_start = (np.maximum(Z_start,0)).tolist()
    	R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
    	Z_stop = (np.minimum(Z_stop,Zs)).tolist()

    	r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
    	z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
    	R[r] = Z[z]
    	return R

# Counting of positive elements of an array.
def counter_positive( x ):
	return sum( 1 for i in x.flatten() if i > 0 )

# Gaussian-like stimulus function.
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
    	return 10.0 * np.exp(-R/2)

# Receptors regular grid. Jitter can be added.
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

# A Gaussian-like function.
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

    	T	= 100    # No of Euler's time discretization
    	dt      = 35.0/float(T)   # Timestep
    	lrate   = 0.05   # Learning rate
    	alpha   = 0.10   # Time constant
    	tau     = 1.00   # Synapse temporal decay
    	epochs  = 5100  # Number of training epochs

    	W_min, W_max = 0.00, 1.00     # Weights min/max values for initialization
    	Ke      = 960.0/(n*n) * 3.65  # Strength of lateral excitatory weights
    	sigma_e = 0.1                 # Extent of lateral excitatory weights
    	Ki      = 960.0/(n*n) * 2.40  # Strength of lateral inhibitory weights
    	sigma_i = 1.0                 # Extent of lateral excitatory weights
    	Kc      = 1.0                 # Strength of field attenuation
    	sigma_c = 2.1                 # Extent of field attenuation

    	# Neural field setup
    	# --------------------------------------------
    	U = np.random.uniform(0.00,0.01,(n,n))
    	V = np.random.uniform(0.00,0.01,(n,n))
    	G = Kc*gaussian((n,n),(sigma_c,sigma_c))

	W = np.load( 'weights10000.npy' )

	# FFT implementation
    	# --------------------------------------------
	We = Ke * alpha * gaussian( (p,p), (0.1,0.1) )
	Wi = Ki * alpha * gaussian( (p,p), (1.0,1.0) )

    	V_shape, We_shape, Wi_shape = np.array(V.shape), np.array(We.shape), np.array(Wi.shape)
    	shape = np.array( best_fft_shape( V_shape + We_shape//2 ) )

	We_fft = rfft2( We[::-1,::-1], shape )
	Wi_fft = rfft2( Wi[::-1,::-1], shape )

	i0 = We.shape[0]//2
	i1 = i0+V_shape[0]
	j0 = We.shape[1]//2
	j1 = j0+V_shape[1]

    	# Skin setup
    	# --------------------------------------------
    	S = np.zeros((Sn,Sn), dtype=np.double)

    	# Receptors setup
    	# --------------------------------------------
    	R = np.zeros((Rn,Rn), dtype=[('x',np.double),('y',np.double)])
    	# R['x'],R['y'] = regular(Rn,noise=R_noise)
    	R['x'] = np.load( 'gridxcoord.npy' )
    	R['y'] = np.load( 'gridycoord.npy' )

    	# Samples generation
    	# --------------------------------------------
    	Sx = np.round(R['x']*(Sn-1)).astype(int)
    	Sy = np.round(R['y']*(Sn-1)).astype(int)
    	samples = np.zeros((Rn*Rn,Rn*Rn)) # 160

	# Building stimuli by applying a lesion.
	# Skin lesion parameters. Lower and upper boound of lesion
	# PLoS article parameters: x: 0.2->0.5 , y: 0.3->0.6
    	lower, upper = 0.15, 0.7
    	ii = 0
    	for i,x in enumerate(np.linspace(-0.75,+0.75,Rn)):
	 	    for j,y in enumerate(np.linspace(-0.75,+0.75,Rn)):
			    S = stimulus((Sn,Sn), width=(0.15,0.15), center = (x,y) )
			    for q in range( Rn ):
				    for s in range( Rn ):
			 		    if ( R['x'][q,s] > lower and R['x'][q,s] < upper and
			 	    	         R['y'][q,s] > lower and R['y'][q,s] < upper ):
			 			 S[Sx[q,s],Sy[q,s]] = 0.0
			    samples[ii] = S[Sx,Sy].ravel()
			    ii += 1
			    # Uncomment the line below just in case of localized lesion
			    #if ( R['x'][q,s] > lower and R['x'][q,s] < upper ):
			           #S[Sx[q,s],Sy[q,s]] = 0.0

    	# Actual training
    	# --------------------------------------------
    	for e in range(epochs):
		# Pick a random sample
        	stimulus = samples[np.random.randint(ii)]

        	# Computes field input accordingly
        	D = (( np.abs( W - stimulus )).sum(axis=-1))/float(Rn*Rn)
        	I = ( 1.0 - D.reshape(n,n) ) * G * alpha

		# Field simulation until convergence
        	for l in range( T ):
			Z = rfft2( V, shape )
			Le = irfft2( Z * We_fft, shape).real[i0:i1,j0:j1]
			Li = irfft2( Z * Wi_fft, shape).real[i0:i1,j0:j1]
            		U += 1.0/tau * dt * ( -U + ( Le - Li ) + I )
            		V = np.maximum(U, 0)

        	# Learning
        	# --------
            	W -= lrate * (Le.ravel() * (W-stimulus).T).T

        	# Field activity reset
        	# --------------------
        	U = np.random.uniform(0.00,0.01,(n,n))
        	V = np.random.uniform(0.00,0.01,(n,n))

		if e%100==0:
			print e
			np.savetxt( os.path.join('D:\Documents\SpyderProjects\Rougier receptive field structure codes (all meant to be the same)\PLoS One (unedited)\SLweights','weights'+str( '%05d' % e )+'.dat'), W )

	np.save( 'SL-weights5000', W )
    	m = Rn
    	plt.figure(figsize=(10,10))
    	ax = plt.subplot(111, aspect=1)
    	R = np.zeros((n*m,n*m))
    	for j in range(n):
	 	for i in range(n):
	 		R[j*m:(j+1)*m,i*m:(i+1)*m] = W[j*n+i].reshape(m,m)
	im = plt.imshow(R, interpolation='nearest', cmap=plt.cm.bone_r,vmin=0,vmax=1)
   	plt.xticks(np.arange(0,n*m,m),[])
    	plt.yticks(np.arange(0,n*m,m),[])

    
    	plt.grid()
	plt.show()
