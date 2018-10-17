import torch
import numpy as np
torch.set_printoptions(8)

def torch_meshgrid(x_inf, x_sup, y_inf, y_sup, xn, yn):
    x = torch.linspace(x_inf, x_sup, xn)[1:]
    y = torch.linspace(y_inf, y_sup, yn)[1:]
    return x.view(1,-1).repeat(yn-1,1), y.view(-1,1).repeat(1,xn-1)
    


if __name__ == '__main__':

#    X, Y = np.meshgrid(np.linspace(x_inf, x_sup, n+1)[1:],
#                       torch.linspace(y_inf, y_sup, n+1)[1:])
    xn = 35
    yn = 22
    x_inf, x_sup, y_inf, y_sup = 0.2, 1.2, 0.0, 2.5
    X,Y = torch_meshgrid(x_inf,x_sup,y_inf,y_sup,xn+1,yn+1)

    Xx, Yy = np.meshgrid(np.linspace(x_inf, x_sup, xn+1)[1:],
                       np.linspace(y_inf, y_sup, yn+1)[1:])