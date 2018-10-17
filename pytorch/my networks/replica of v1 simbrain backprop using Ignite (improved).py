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
from torch.utils.data import DataLoader, Dataset
import pdb
from pprint import pprint
import datetime

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.DoubleTensor')

#now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#summary_writer = SummaryWriter(log_dir=f"tf_log/exp_ignite_{now}")

class AbridgingDataset(Dataset):
    """ Abridging dataset. """
    
    # Initialize your data, download, etc.
    def __init__(self):
#        x = np.genfromtxt('data/double/annie_double_ordered_INPUT_data.csv', delimiter=',', dtype=np.float32, skip_header=1)
#        y = np.genfromtxt('data/double/annie_double_ordered_TARGET_data.csv', delimiter=',', dtype=np.float32, skip_header=1)
        torch_x = torch.tensor(pd.read_csv('data/annie_long_double_training_input_data.csv').values, dtype=torch.float32)
        torch_y = torch.tensor(pd.read_csv('data/annie_long_double_training_target_data.csv').values, dtype=torch.float32)
#        torch_x = torch.tensor(pd.read_csv('data/single/rachel_single_ordered_INPUT_data.csv').values, dtype=torch.float32)
#        torch_y = torch.tensor(pd.read_csv('data/single/rachel_single_ordered_TARGET_data.csv').values, dtype=torch.float32)
#        torch_x = torch.tensor(pd.read_csv('data/double/annie_double_ordered_INPUT_data.csv').values, dtype=torch.float32)
#        torch_y = torch.tensor(pd.read_csv('data/double/annie_double_ordered_TARGET_data.csv').values, dtype=torch.float32)


        self.len = torch_x.shape[0]
        self.x_norm = nn.functional.normalize(torch_x).double()
        self.y_norm = nn.functional.normalize(torch_y).double()
#        self.x_data = torch.tensor(pd.read_csv('data/double/annie_double_ordered_INPUT_data.csv').values, dtype=torch.float32).view(144,3)
#        self.y_data = torch.tensor(pd.read_csv('data/double/annie_double_ordered_TARGET_data.csv').values, dtype=torch.float32).view(144,1)
        
    def __getitem__(self, index):
        return self.x_norm[index], self.y_norm[index]
    
    def __len__(self):
        return self.len

#torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#writer = SummaryWriter()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
#N, D_in, H, D_out = 64, 1000, 100, 10
N, D_in, H, D_out = 24, 3, 100, 1

from ignite.metrics import (
        CategoricalAccuracy, 
        Loss,
        Precision,
    )

from ignite.engine import (
        create_supervised_evaluator,
        create_supervised_trainer,
        Events,
    )


#x_norm = nn.functional.normalize(torch_x)
#y_norm = nn.functional.normalize(torch_y)

dataset = AbridgingDataset()
train_dl = DataLoader(dataset=dataset,
                      batch_size=24,
                      shuffle=True,
#                      num_workers=2,
)

#val_dl = DataLoader(
#        y_norm,
#        batch_size=24,
#        shuffle=False,
#        num_workers=4,
#)


# Use the nn package to define our model and loss function.
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
).to(device)
#pdb.set_trace()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss(reduction='elementwise_mean')
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(
#        model.parameters(),
#        lr = 0.001,
#        momentum=0.9,
#)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.

#def training_update_function(batch):
#    model.train()
#    optimizer.zero_grad()
#    input, target = input_dl, target_dl
#    output = model(input)
#    loss = criterion(output, target)
#    loss.backward()
#    optimizer.step()
#    return loss.data[0]
#
#trainer = Trainer(train_dl, training_update_function)
#trainer.run(num_epochs=5)
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(
    model,
    metrics={
        "accuracy": CategoricalAccuracy(),
        "loss": Loss(criterion),
        "precision": Precision(),
    },
    device=device,
)

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}] Batch[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
#    
#@trainer.on(Events.EPOCH_COMPLETED)
#def log_training_results(trainer):
#    evaluator.run(train_dl)
#    metrics = evaluator.state.metrics
#    print(f"Training Results   - Epoch: {trainer.state.epoch}  "
#          f"accuracy: {metrics['accuracy']} "
#          f"loss: {metrics['loss']} "
#          f"prec: {metrics['precision'].cpu()}")
    
#@trainer.on(Events.EPOCH_COMPLETED)
#def log_validation_results(engine):
#    evaluator.run(val_dl)
#    metrics = evaluator.state.metrics
#    print(f"Training Results   - Epoch: {trainer.state.epoch}  "
#          f"loss: {metrics['loss']:.2f} "
#          f"prec: {metrics['precision'].cpu()}")

trainer.run(train_dl, 10)
#trainer.run(train_dl, 100)
#            



















#for t in range(20000):
#    # Forward pass: compute predicted y by passing x to the model.
#    y_pred = model(x_norm)
#
#    # Compute and print loss.
#    loss = loss_fn(y_pred, y_norm)
#    print(t, loss.item())
#
#    optimizer.zero_grad()
#
#    loss.backward()
#
#    optimizer.step()
##    with torch.no_grad():
##        for param in model.parameters():
##            param -= learning_rate * param.grad
