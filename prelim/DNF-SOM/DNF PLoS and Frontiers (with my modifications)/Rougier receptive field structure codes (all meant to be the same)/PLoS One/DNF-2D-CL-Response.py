'''
Model response. Calculates the response of the DNF-SOM-MultiDimInput
model.
'''
import math as mt
import numpy as np
from sys import stdout
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2

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

def g(x, sigma=1.0):
    return np.exp(-0.5*(x/sigma)**2)

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
				print 'Wrong start point!'
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
				print 'Wrong start point!'
				return -1
			else:
				length = int( length )
				silence[begin:begin+length,begin:begin+length] = 0.0
				return silence

def area_of_activity( data ):
	return sum( 1 for i in data.flatten() if i > 0.0 )

# Receptors regular grid. Jitter can be added.
def grid(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, noise=0.0):
	x = np.linspace(xmin,xmax,n)
    	y = np.linspace(ymin,ymax,n)
    	X,Y = np.meshgrid(x,y)
    	X += np.random.uniform(-noise, noise, (n,n))
    	Y += np.random.uniform(-noise, noise, (n,n))
    	return X.ravel(), Y.ravel()

def dnfsom_activity( n, Rn, l, tau, T, alpha ):
	ms = 0.001	# ms definition
	dt = 100.0 * ms	# Euler's time step
	p = 2*n+1

    	# Building lesion masks
    	# lesion  = cortical_lesion( n, Rn, 15, 2, 10 )
    	# silence = cortical_silence( n, 15, 2, 10 )
    	lesion  = cortical_lesion( n, Rn, 20, 1, 10  )
    	silence = cortical_silence( n, 20, 1, 10 )
    	# lesion  = cortical_lesion( n, Rn, 20, 1, 13 )
    	# silence = cortical_silence( n, 20, 1, 13 )

	# Allocation of arrays and loading necessary files
	O = np.zeros((l*n,l*n))
	W = np.load( 'weights10000.npy' )
	W *= lesion
	V = np.random.random((n,n)) * .01
	U = np.random.random((n,n)) * .01
	U *= silence
	V *= silence
	R = np.zeros((Rn*Rn, 2))
	R[:,0] = np.load( 'gridxcoord.npy' )
	R[:,1] = np.load( 'gridycoord.npy' )

	# FFT implementation
	We = alpha * 3.65 * 960.0/(n*n) * gaussian( (p,p), (0.1,0.1) )
	Wi = alpha * 2.40 * 960.0/(n*n) * gaussian( (p,p), (1.0,1.0) )
	sigma_c = 2.10
	G = gaussian( (n,n), (sigma_c,sigma_c) )

    	V_shape, We_shape, Wi_shape = np.array(V.shape), np.array(We.shape), np.array(Wi.shape)
    	shape = np.array( best_fft_shape( V_shape + We_shape//2 ) )

	We_fft = rfft2( We, shape )
	Wi_fft = rfft2( Wi, shape )

	i0 = We.shape[0]//2
	i1 = i0+V_shape[0]
	j0 = We.shape[1]//2
	j1 = j0+V_shape[1]

	# Stimuli generation
	size = l*l
	S = np.zeros((size,2))
	S[:,0], S[:,1] = grid(l,0.1, 0.9, 0.1, 0.9, 0.0)
    	dX = np.abs(R[:,0].reshape(1,Rn*Rn) - S[:,0].reshape(size,1))
    	dY = np.abs(R[:,1].reshape(1,Rn*Rn) - S[:,1].reshape(size,1))
    	samples = np.sqrt(dX*dX+dY*dY)
    	samples = g(samples, 0.08)

	# Calculation of model response
	step = 0
	jj   = 100.0/( float(l) )
	for i in range( l ):
		for j in range( l ):
			D = (( np.abs( W - samples[i*l+j] )).sum(axis=-1))/float(Rn*Rn)
			I = ( 1.0 - D.reshape(n,n) ) * G * alpha

			for k in range( int(T/dt) ):
				Z = rfft2( V, shape )
				Le = irfft2( Z * We_fft, shape ).real[i0:i1,j0:j1]
				Le *= silence
				Li = irfft2( Z * Wi_fft, shape ).real[i0:i1,j0:j1]
				Li *= silence
				U += ( -U + ( Le - Li ) + I ) * tau * dt
				U *= silence
				V = np.maximum( U, 0 )
				V *= silence

			O[i*n:(i+1)*n,j*n:(j+1)*n] = V

			V = np.random.random((n,n)) * .01
			U = np.random.random((n,n)) * .01
			U *= silence
			V *= silence

		step += jj
		progress_bar( 30, step )
	np.save( 'model_response_64_cl', O )

if __name__ == '__main__':
	model_size = 32
	rf_resolution = 64
	num_receptors = 16
	T = 10.0
	tau = 1.0
	alpha = 0.1

	dnfsom_activity( model_size, num_receptors, rf_resolution, tau, T, alpha )
