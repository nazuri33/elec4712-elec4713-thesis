SITopMap (Somatosensory cortex Topographic Maps) is the implementation in 
Python of the model described in:
"A Neural Field Model of the Somatosensory Cortex: Formation, Maintenance 
and Reorganization of Ordered Topographic Maps, Georgios Is. Detorakis and 
Nicolas P. Rougier, PLoS ONE DOI: 10.1371/journal.pone.0040257"

The present tarball contains three Python script files:
- DNF-2D-SOM-REF.py : Is the main script and implements the model. The user 
  can get the feed-forward weights.

- DNF-2D-REF-Response.py : This script computes the receptive fields of the 
  model for a given stimuli set.

- DNF-RF-Size.py : This script computes and plots all the receptive fields 
  of the model.

In addition, a precomputed set of feed-forward weights (weights.npy) has been
included in the present tarball, as well as the corresponding receptors 
coordinates (gridxcoord.npy and gridycoord.npy), and a model response of a 
resolution of 64 pixels (model_response_64).
