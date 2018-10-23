'''
Model response. Calculates the response of the DNF-SOM-MultiDimInput
model.
'''
import os, sys
import math as mt
import numpy as np
from sys import stdout
from time import sleep
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, rfft2, irfft2, fftshift, ifftshift

def progress_bar( width, percent ):
	marks  = mt.floor( width * ( percent/100.0 ) )
	spaces = mt.floor( width - marks )
	loader = '[' + ('#' * int( marks) ) + (' ' * int(spaces) ) + ']'

	stdout.write("%s %d%%\r" % (loader, percent) )
	if percent >= 100:
		stdout.write("\n")
	stdout.flush()

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

def gaussian(shape=(25,25), width=(1,1), center=(0,0)):
	grid=[]
	for size in shape:
		grid.append (slice(0,size))
	C = np.mgrid[tuple(grid)]
	R = np.zeros(shape)
	for i,size in enumerate(shape):
		if shape[i] > 1:
			R += (((C[i]/float(size-1))*2 - 1 - center[i])/width[i])**2
	return np.exp(-R/2)

def sigmoid( x, s, th ):
	return 1.0/( 1.0 + np.exp( -s*( x - th ) ) )

def dnfsom_activity( n, m, stimulus, W, tau, T, alpha ):
	p = 2 * n + 1
	dt = 35.0/float(T)
	V = np.random.random((n,n)) * .01
	U = np.random.random((n,n)) * .01

	We = alpha * 1.50 * 960.0/(n*n) * gaussian( (p,p), (0.1,0.1) )
	Wi = alpha * 0.75 * 960.0/(n*n) * gaussian( (p,p), (1.0,1.0) )
	sigma_c = 2.10
	G = gaussian( (n,n), (sigma_c,sigma_c) )

    	V_shape, We_shape, Wi_shape = np.array(V.shape), np.array(We.shape), np.array(Wi.shape)
    	shape = np.array( best_fft_shape( V_shape + We_shape//2 ) )

	We_fft = rfft2( We[::-1,::-1], shape )
	Wi_fft = rfft2( Wi[::-1,::-1], shape )

	i0 = We.shape[0]//2
	i1 = i0+V_shape[0]
	j0 = We.shape[1]//2
	j1 = j0+V_shape[1]

	D = (( np.abs( W - stimulus )).sum(axis=-1))/float(m*m)
	I = ( 1.0 - D.reshape(n,n) ) * G * alpha

	for i in range( T ):
		Z = rfft2( V, shape )
		Le = irfft2( Z * We_fft, shape ).real[i0:i1,j0:j1]
		Li = irfft2( Z * Wi_fft, shape ).real[i0:i1,j0:j1]
		U += ( -U + ( Le - Li ) + I ) * 1.0/tau * dt
		V = np.maximum( U, 0 )
	return V

if __name__ == '__main__':
	n = 32
	p = 2 * n + 1
	m = 16
	l = 64
	T = 100
	tau = 1.0
	alpha = 0.1

	W = np.genfromtxt('SLweights05000.dat')#W  = np.load( 'SL-weights5000.npy' ) #W = np.random.uniform(W_min,W_max,(n*n,Rn*Rn))
 	Rx = np.load( 'gridxcoord.npy' )
 	Ry = np.load( 'gridycoord.npy' )

	Sn = 64
	Rn = m
	Sx = np.round(Rx*(Sn-1)).astype(int)
    	Sy = np.round(Ry*(Sn-1)).astype(int)
	O  = np.zeros((l*n,l*n))
    	samples = np.zeros((l*l,Rn*Rn))

	ii   = 0
	step = 0
	jj   = 100.0/( float(l) )
	for i,x in enumerate(np.linspace(-0.75,+0.75,l)):
		for j,y in enumerate(np.linspace(-0.75,+0.75,l)):
			S = stimulus((Sn,Sn), width=(0.15,0.15), center = (x,y) )
			samples[ii] = S[Sx,Sy].ravel()
			Z = dnfsom_activity( n, m, samples[ii], W, tau, T, alpha )
	    		O[i*n:(i+1)*n,j*n:(j+1)*n] = Z
			ii += 1
		step += jj
		progress_bar( 30, step )

	np.save( 'generated5000SLweights2D_response_64', O )
