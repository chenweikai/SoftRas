import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np


class SphericalHarmonicsLighting(nn.Module):
    # using 9 component spherical harmonic coefficents for each color
    # based on representation in [1] "An Efficient Representation for Irradiance Environment Maps" by Ravi Ramamoorthi and Pat Hanrahan
    def __init__(self):
        super(SphericalHarmonicsLighting, self).__init__()

        # constants from eq. 12 in [1]
        self.c_ = np.array([0.429043, 0.511664, 0.743125, 0.886227, 0.247708])
        from scipy.sparse import coo_matrix
        rows = np.array(list(range(0, 16)) + [15])
        print(rows)
        cols = np.array(
            [8, 4, 7, 3, 4, 8, 5, 1, 7, 5, 6, 2, 3, 1, 2, 0, 6])
        vals = np.array([self.c_[0], self.c_[0], self.c_[0], self.c_[1]]
                        + [self.c_[0], -self.c_[0], self.c_[0], self.c_[1]]
                        + [self.c_[0], self.c_[0], self.c_[2], self.c_[1]]
                        + [self.c_[1], self.c_[1], self.c_[1], self.c_[3], -self.c_[4]])
        print(vals)
        T_np_sparse = coo_matrix((vals, (rows, cols)), shape=(16, 9))
        T_np = T_np_sparse.toarray()
        self.T_ = Variable(torch.FloatTensor(T_np), requires_grad=False)
        self.dtype_ = torch.FloatTensor


def test():
    shLight = SphericalHarmonicsLighting()


if __name__ == '__main__':
    a = np.floor(10*np.random.randn(3,4))
    print(type(a))
    b = a.ravel()
    print(b)
    c = b.reshape(6,2)
    print(c)
    #test()
