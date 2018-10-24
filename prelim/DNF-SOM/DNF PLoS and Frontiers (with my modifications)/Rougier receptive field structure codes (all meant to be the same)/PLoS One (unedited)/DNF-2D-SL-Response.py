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

def area_of_activity( data ):
	return sum( 1 for i in data.flatten() if i > 0.0 )

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

# Receptors regular grid. Jitter can be added.
def grid(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, noise=0.0):
	x = np.linspace(xmin,xmax,n)
	y = np.linspace(ymin,ymax,n)
	X,Y = np.meshgrid(x,y)
	X += np.random.uniform(-noise, noise, (n,n))
	Y += np.random.uniform(-noise, noise, (n,n))
	return X.ravel(), Y.ravel()

def plot_samples_sum( data, Rn, size ):
	t = np.zeros((Rn,Rn))
	for i in range( size ):
		t += data[i].reshape(Rn,Rn)
	plt.imshow( t )
	plt.colorbar()
	plt.show()

def dnfsom_activity( n, Rn, l, tau, T, alpha ):
	ms = 0.001	# ms definition
	dt = 100.0 * ms	# Euler's time step
	p = 2*n+1

	# Allocation of arrays and loading necessary files
	O = np.zeros((l*n,l*n))
	W = np.genfromtxt( 'SLweights05000.dat' )
	V = np.random.random((n,n)) * .01
	U = np.random.random((n,n)) * .01

	# FFT implementation
	We = alpha * 3.75 * 960.0/(n*n) * gaussian( (p,p), (0.1,0.1) )
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

	# Samples generation
    	# --------------------------------------------
	R = np.zeros((Rn*Rn, 2))
	R[:,0] = np.load( 'gridxcoord.npy' ).ravel()
	R[:,1] = np.load( 'gridycoord.npy' ).ravel()

	mask = np.ones((Rn,Rn))
	mask[:,6:10] = 0.0
	mask = mask.ravel()
	R[:,0] *= mask
	R[:,1] *= mask

	size = l*l
	S = np.zeros((l*l,2))
	S[:,0], S[:,1] = grid( l, 0.1, 0.9, 0.1, 0.9, 0.0 )
    	dX = np.abs(R[:,0].reshape(1,Rn*Rn) - S[:,0].reshape(size,1))
    	dY = np.abs(R[:,1].reshape(1,Rn*Rn) - S[:,1].reshape(size,1))
    	samples = np.sqrt(dX*dX+dY*dY)
    	samples = 3. * g(samples, 0.08)

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
				Li = irfft2( Z * Wi_fft, shape ).real[i0:i1,j0:j1]
				U += ( -U + ( Le - Li ) + I ) * tau * dt
				V = np.maximum( U, 0 )
			O[i*n:(i+1)*n,j*n:(j+1)*n] = V

			V = np.random.random((n,n)) * .01
			U = np.random.random((n,n)) * .01
		step += jj
		progress_bar( 30, step )

	plt.imshow( O, interpolation='bicubic', cmap=plt.cm.hot, extent = [0,l*n,0,l*n])
    	plt.xticks(np.arange(0,l*n,n),[])
    	plt.yticks(np.arange(0,l*n,n),[])

	np.save( 'given_skin_SOM_weights_64_sl', O )
	plt.show()

if __name__ == '__main__':
	model_size = 32
	rf_resolution = 64
	num_receptors = 16
	T = 10.0
	tau = 1.0
	alpha = 0.1

	dnfsom_activity( model_size, num_receptors, rf_resolution, tau, T, alpha )
