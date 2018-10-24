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
""" This script computes and plots the center of mass of each receptive field
    of the model and its size. """
import numpy as np
import matplotlib.pylab as plt


def area_of_activation(data):
    """ It computes the area of activation of the neural field
    ** Parameters **

    data : double
        The activity of the model (neural field)
    """
    return sum(1 for i in data.flatten() if i > 0.0)


def rf_size(Input, net_size, resolution):
    """ It computes the center of mass of each receptive fields and its size
        as well.
    **Parameters**

    Input : double
        The raw data (activity of the neural field)

    net_size : int
        The size of the model (neural field)

    resolution : int
        The size of a receptive field
    """
    size = np.zeros((net_size*net_size,))
    coms = np.zeros((net_size*net_size, 2))
    I = np.zeros((net_size, net_size, resolution, resolution))
    R = np.zeros((net_size*resolution, net_size*resolution))
    Z = np.zeros((resolution, resolution))

    scale = 1.0/(resolution**2)

    X, Y = np.meshgrid(np.arange(Z.shape[0]), np.arange(Z.shape[1]))

    count_roi, count_nroi, count_tot = 0, 0, 0
    for i in range(net_size):
        for j in range(net_size):
            I[i, j, ...] = Input[i::net_size, j::net_size]
            Z = I[i, j, ...]
            R[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution] = Z
            size[i*net_size+j] = area_of_activation(Z) * scale

            d = np.unravel_index(Z.argmax(), Z.shape)
            Z = np.roll(Z, Z.shape[0]/2-d[0], axis=0)
            Z = np.roll(Z, Z.shape[1]/2-d[1], axis=1)

            xc = (((Z*Y).sum() / Z.sum() - Z.shape[0]/2 +
                   d[0])/float(Z.shape[0]))
            yc = (((Z*X).sum() / Z.sum() - Z.shape[1]/2 +
                   d[1])/float(Z.shape[1]))

            coms[i*net_size+j, 0] = (xc + 1.0) % 1
            coms[i*net_size+j, 1] = (yc + 1.0) % 1
    return coms, R, size


def plot_rfs(size, C, Rx, Ry):
    """ It plots the receptive fields of the model.
    **Parameters**

    size : int (array)
        It contains the size of each receptive field

    C : double (array)
        It contains the centers of masses of the receptive fields

    Rx, Ry : double
        The x, y coordinates of the receptors
    """
    radius = np.sqrt(size[...]/np.pi)
    plt.scatter(Rx, Ry, s=15, color='w', edgecolor='k')
    plt.scatter(C[..., 1], C[..., 0], s=radius*1200, alpha=0.4, color='b')
    plt.xticks([])
    plt.yticks([])


if __name__ == '__main__':
    net_size = 32
    resolution = 64

    Rx = np.load('gridxcoord.npy')
    Ry = np.load('gridycoord.npy')
    O = np.load('model_response_64.npy')
    C, R, size = rf_size(O, net_size, resolution)

    print sum(np.isnan(C[..., 0]))

    plt.figure(figsize=(9, 9))
    plot_rfs(size, C, Rx, Ry)
    plt.show()
