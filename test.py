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

    import torch
    a = torch.FloatTensor(np.arange(16).reshape(4,4))
    # print(a)
    normal = torch.FloatTensor(np.random.randn(4,1))
    normal_T = normal.transpose(0,1)

    # single_output = torch.mm(a, normal)
    # single_final = torch.mm(normal_T, single_output)
    # print(single_output) 
    # print(single_final)
    # print(normal)
    # print(normal.size())

    
    # print(normal)
    # print(normal.size())
    batch_size = 5
    print(normal)
    normal = normal[None, :, :]
    normal_exp = normal.expand(2, normal.size(1), normal.size(2))
    print(normal_exp)
    print(normal.size())
    print(normal_exp.size())
    # normal_T = normal_T[None, :, :].repeat(batch_size, 1, 1,)
    # c = normal.repeat(batch_size, 1, 1)
    # # print(normal)
    # # print(c)
    # # print(c.size())

    # a_ = a[None, :, :].repeat(batch_size, 1, 1)
    # b_ = a[None, :, :].expand()
    # output = torch.bmm(a_, c)
    # print(output)
    # output_final = torch.bmm(normal_T, output)
    # print(output_final)
    # print(output_final.size())
    # output_final.squeeze_()
    # print(output_final.size())
    
