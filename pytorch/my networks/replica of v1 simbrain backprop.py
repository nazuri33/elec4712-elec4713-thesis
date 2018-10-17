# -*- coding: utf-8 -*-
"""
PyTorch: optim
--------------

A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.

Rather than manually updating the weights of the model as we have been doing,
we use the optim package to define an Optimizer that will update the weights
for us. The optim package defines many optimization algorithms that are commonly
used for deep learning, including SGD+momentum, RMSProp, Adam, etc.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import tensorboardX as SummaryWriter

torch.set_default_tensor_type('torch.FloatTensor')
writer = SummaryWriter()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
#N, D_in, H, D_out = 64, 1000, 100, 10
N, D_in, H, D_out = 24, 3, 6, 1

# Create random Tensors to hold inputs and outputs
#x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)
#
#x = torch.load('data/double/annie_double_ordered_INPUT_data.csv')
#y = torch.load('data/double/annie_double_ordered_TARGET_data.csv')
#
#x = pd.read_csv('data/double/annie_double_ordered_INPUT_data.csv')
#y = pd.read_csv('data/double/annie_double_ordered_TARGET_data.csv')

#nump_x = np.genfromtxt('data/double/annie_double_ordered_INPUT_data.csv', delimiter=",", skip_header=1)
#nump_y = np.genfromtxt('data/double/annie_double_ordered_TARGET_data.csv', delimiter=",", skip_header=1)
#torch_x = torch.from_numpy(nump_x)
#torch_y = torch.from_numpy(nump_y).view(144,1)
torch_x = torch.tensor(pd.read_csv('data/double/annie_double_ordered_INPUT_data.csv').values, dtype=torch.float32).view(144,3)
torch_y = torch.tensor(pd.read_csv('data/double/annie_double_ordered_TARGET_data.csv').values, dtype=torch.float32).view(144,1)
x_norm = nn.functional.normalize(torch_x)
y_norm = nn.functional.normalize(torch_y)

# Use the nn package to define our model and loss function.
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
)
loss_fn = nn.MSELoss(reduction='elementwise_mean')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(20000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_norm)

    # Compute and print loss.
    loss = loss_fn(y_pred, y_norm)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
#    with torch.no_grad():
#        for param in model.parameters():
#            param -= learning_rate * param.grad
