import numpy as np
import matplotlib.pylab as plt
# import scipy.spatial.distance as sptd
# import cogent.maths.stats.test as stats

def counts_positive( x ):
	return sum( 1 for i in x.flatten() if i > 0 )

def rf_size( Input, net_size, resolution ):
	size = np.zeros((net_size*net_size,))
	I = np.zeros((net_size,net_size,resolution,resolution))
    	R = np.zeros((net_size*resolution,net_size*resolution))
	C = np.zeros((net_size*net_size,2))
	scale = 1.0/( resolution**2 )

	X,Y = np.mgrid[0:resolution,0:resolution]
	X = X/float( resolution - 1 )
	Y = Y/float( resolution - 1 )

	ii = 0
	for k in range( net_size ):
		for q in range( net_size ):
			I[k,q,...] = Input[k::32,q::32]
			R[k*resolution:(k+1)*resolution,q*resolution:(q+1)*resolution] = I[k,q,...]
			size[ii] = sum( 1 for v in I[k,q,...].flatten() if v > 0 ) * scale
			C[ii,0] = ( X * I[k,q,...] ).sum()/I[k,q,...].sum()
			C[ii,1] = ( Y * I[k,q,...] ).sum()/I[k,q,...].sum()
			ii += 1
	return C, R, size

def plot_statistics( data, bins ):
	plt.hist( data, bins )
	plt.axis([0,0.1,0,100])

def plot_rfs( data, C, Rx, Ry ):
	radius = np.sqrt( data[...]/np.pi )
	scale_f = 1.5

	plt.scatter( Rx, Ry, s=15, color='w', edgecolor='k' )
	plt.scatter( ( C[...,1] ),
	             ( C[...,0] ),
		       s=radius*1200, alpha=0.4, color='b' )
	plt.xticks([])
	plt.yticks([])

def main():
	net_size = 32
	resolution = 64

	print 'Loading necessary files...'
	Rx = np.load( 'gridxcoord.npy' )
	Ry = np.load( 'gridycoord.npy' )
	O  = np.load( 'model_response_64_sl_PLoS1weights.npy' )
	print '...Ok!'

	print 'Calculating RFs sizes and locations...'
	C, R, size = rf_size( O, net_size, resolution )
	print '...Ok!'

	print 'Visualizing the results...'
	plt.figure( figsize=(10,10) )
	plot_rfs( size, C, Rx, Ry )
	plt.savefig( 'rfs-locations.png' )

	plt.figure( figsize=(7,7) )
	plot_statistics( size, 100 )
	# plt.savefig( 'rfs-histogram.png' )
	print '...Done!'

	print 'Number of dead neurons:',sum( np.isnan( C[...,0] ) )
	plt.show()

if __name__ == '__main__':
	main()
