import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

import soft_renderer.functional as srf

def batch_row_wise_vector_product(in_1, in_2):
    # in_1: batch_size * V_num * 4; in_2: batch_size * 4 * V_num
    # each vertex has an augmented normal (x, y, z, 1)
    # we want to compute the dot product for each vertex
    # hence the output size is: batch_size * V_num
    dim = in_1.size(0)
    tmp = []
    for i in range(dim):
        tmp_1 = in_1[i, :, :].squeeze_()
        tmp_2 = in_2[i, :, :].squeeze_()
        tmp.append(row_wise_vector_product(tmp_1, tmp_2))
    output = torch.stack(tmp, 0)
    return output


def row_wise_vector_product(var_1, var_2, dim=0):
    # temporal pytorch util
    # compute row/col-wise vector product
    # assume input are 2D pytorch variable / tensor
    # var_1: M * N; var_2: N * M
    # if dim==0, output size is M; if dim==1, output size is N
    if dim == 0:
        d = var_1.size(1)
        E = torch.bmm(var_1.view(-1, 1, d), var_2.view(-1, d, 1)).squeeze()
    elif dim == 1:
        d = var_1.size(0)
        E = torch.bmm(var_1.transpose(0, 1).contiguous().view(-1, 1, d),
                      var_2.transpose(0, 1).contiguous().view(-1, d, 1)).squeeze()
    else:
        E = None
    return E

class SphericalHarmonicsLighting(nn.Module):
    # using 9 component spherical harmonic coefficents for each color
    # based on representation in [1] "An Efficient Representation for Irradiance Environment Maps" by Ravi Ramamoorthi and Pat Hanrahan
    def __init__(self, sh_coeffs):
        super(SphericalHarmonicsLighting, self).__init__()

        # constants from eq. 12 in [1]
        self.c_ = np.array([0.429043, 0.511664, 0.743125, 0.886227, 0.247708])
        self.sh_coeffs = sh_coeffs

        # transform for the transform matrix "M" as in eq. 11 in [1]
        from scipy.sparse import coo_matrix
        rows = np.array(list(range(0, 16)) + [15])
        cols = np.array([8, 4, 7, 3, 4, 8, 5, 1, 7, 5, 6, 2, 3, 1, 2, 0, 6])
        vals = np.array([self.c_[0], self.c_[0], self.c_[0], self.c_[1]]
                        + [self.c_[0], -self.c_[0], self.c_[0], self.c_[1]]
                        + [self.c_[0], self.c_[0], self.c_[2], self.c_[1]]
                        + [self.c_[1], self.c_[1], self.c_[1], self.c_[3], -self.c_[4]])

        T_np_sparse = coo_matrix((vals, (rows, cols)), shape=(16, 9))
        T_np = T_np_sparse.toarray()
        # self.T_ = Variable(torch.FloatTensor(T_np), requires_grad=False)
        self.T_ = torch.FloatTensor(T_np)
        self.dtype_ = torch.cuda.FloatTensor

    def compute_irradiance_transfrom(self, sh_coeffs):
        # create transform matrix "M" as in eq. 11 in [1]
        # note: only for one color
        shCoeffs = torch.FloatTensor(sh_coeffs).cuda()
        M = torch.mm(self.T_.cuda(), shCoeffs.view(-1, 1))
        M2 = M.view(4, 4)
    
        return M2

    def compute_irradiance(self, normals_aug, sh_coeffs):
        # evaluate eq. 11 in [1], for one color
        # sh_coeffs in size (9,) for one color
        # assume normals_aug, size (batch_size, V,4), are normals in camera coordinates

        M = self.compute_irradiance_transfrom(sh_coeffs)
        M = M[None, :, :] # 1 * 4 * 4
        batch_size = normals_aug.size(0)
        M = M.expand(batch_size, M.size(1), M.size(2)) # batch_size * 4 * 4
        N = normals_aug # batch_size * V_num * 4
        MN = torch.bmm(M, N.transpose(1, 2))  # batch_size * 4 * V_num

        E = batch_row_wise_vector_product(N, MN)

        # MN_t = MN.transpose(1, 2).contiguous()       
        # E = row_wise_vector_product(MN_t, N, dim=0)

        return E

    def forward(self, light, normals):
        # normals, size (batch_size, V,3), are normals in camera coordinates
        # sh_coeffs, size (27,), coefficients for r, g, b

        # augment the normals

        device = light.device
        batch_size = normals.size(0)
        v_num = normals.size(1)
        ones = torch.ones(v_num, 1, requires_grad=False).cuda()
        ones = ones[None, :, :]
        normals_aug = torch.cat((normals, ones), dim=2)

        # compute irradiance
        E_red = self.compute_irradiance(normals_aug, self.sh_coeffs[0:9])
        E_green = self.compute_irradiance(normals_aug, self.sh_coeffs[9:18])
        E_blue = self.compute_irradiance(normals_aug, self.sh_coeffs[18:27])
        E_rgb = torch.cat(
            (E_red.view(batch_size, -1, 1), E_green.view(batch_size, -1, 1), E_blue.view(batch_size, -1, 1)), dim=2)

        # compute radiosity - disable radiosity computation for now
        # B = torch.mul( per_vertex_albedo.view(-1,3), E_rgb )
        # B = per_vertex_albedo.view(-1, 3) * E_rgb

        # todo: add clipping function (constrain radiosity range to [0,1])
        # torch.clamp(input, min, max, out=None) -> Tensor

        return E_rgb

class AmbientLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1)):
        super(AmbientLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color

    def forward(self, light):
        return srf.ambient_lighting(light, self.light_intensity, self.light_color)


class DirectionalLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0)):
        super(DirectionalLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color
        self.light_direction = light_direction

    def forward(self, light, normals):
        return srf.directional_lighting(light, normals,
                                        self.light_intensity, self.light_color, 
                                        self.light_direction)


class Lighting(nn.Module):
    def __init__(self, light_mode='surface',
                 intensity_ambient=0.5, color_ambient=[1,1,1],
                 intensity_directionals=0.5, color_directionals=[1,1,1],
                 directions=[0,1,0],
                 sh_coeffs = [0.79, 0.39, -0.34, -0.29, -0.11, -0.26, -0.16, 0.56, 0.21,
                 0.44, 0.35, -0.18, -0.06, -0.05, -0.22, -0.09, 0.21, -0.05, 
                 0.54, 0.60, -0.27, 0.01, -0.12, -0.47, -0.15, 0.14, -0.30] # sh coeffs for Grace Cathedral env map
                 ):
        super(Lighting, self).__init__()

        if light_mode not in ['surface', 'vertex', 'sh']:
            raise ValueError('Lighting mode only support surface, vertex and spherical harmonics')

        self.light_mode = light_mode
        self.ambient = AmbientLighting(intensity_ambient, color_ambient)
        self.directionals = nn.ModuleList([DirectionalLighting(intensity_directionals,
                                                               color_directionals,
                                                               directions)])
        self.sh = SphericalHarmonicsLighting(sh_coeffs)                                                

    def forward(self, mesh):
        if self.light_mode == 'surface':
            light = torch.zeros_like(mesh.faces, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.ambient(light)
            for directional in self.directionals:
                light = directional(light, mesh.surface_normals)
            mesh.textures = mesh.textures * light[:, :, None, :]

        elif self.light_mode == 'vertex':
            light = torch.zeros_like(mesh.vertices, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.ambient(light)
            for directional in self.directionals:
                light = directional(light, mesh.vertex_normals)
            mesh.textures = mesh.textures * light

        elif self.light_mode == 'sh':
            light = torch.zeros_like(mesh.faces, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.sh(light, mesh.vertex_normals)
            mesh.textures = mesh.textures * light#[:, :, None, :]
            # print(mesh.textures)
            # clip and normalize to [0, 1]
            mesh.textures = torch.clamp(mesh.textures, min=0.0)
            # print(mesh.textures)
            max_v = torch.max(mesh.textures)
            # print(max_v)
            scale = 1.0 / max_v
            mesh.textures = mesh.textures * scale
            # print(mesh.textures)

        return mesh
