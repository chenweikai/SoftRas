import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import soft_renderer.functional as srf


def row_wise_vector_product(var_1, var_2, dim=0):
    # temporal pytorch util
    # compute row/col-wise vector product
    # assume input are 2D pytorch variable / tensor
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
        self.T_ = Variable(torch.FloatTensor(T_np), requires_grad=False)
        self.dtype_ = torch.FloatTensor

    def compute_irradiance_transfrom(self):
        # create transform matrix "M" as in eq. 11 in [1]
        # note: only for one color

        M = torch.mm(self.T_, self.sh_coeffs.view(-1, 1))
        M2 = M.view(4, 4)
        # Variable( torch.zeros(4,4), requires_grad=False ) # todo: require grad?
        # M[0][0] = self.c_[0] * sh_coeffs[8];  M[0][1] = self.c_[0] * sh_coeffs[4]
        # M[1][0] = self.c_[0] * sh_coeffs[4];  M[1][1] = -self.c_[0] * sh_coeffs[8]
        # M[2][0] = self.c_[0] * sh_coeffs[7];  M[2][1] = self.c_[0] * sh_coeffs[5]
        # M[3][0] = self.c_[1] * sh_coeffs[3];  M[3][1] = self.c_[1] * sh_coeffs[1]
        # M[0][2] = self.c_[0] * sh_coeffs[7];  M[0][3] = self.c_[1] * sh_coeffs[3]
        # M[1][2] = self.c_[0] * sh_coeffs[5];  M[1][3] = self.c_[1] * sh_coeffs[1]
        # M[2][2] = self.c_[2] * sh_coeffs[6];  M[2][3] = self.c_[1] * sh_coeffs[2]
        # M[3][2] = self.c_[1] * sh_coeffs[2];  M[3][3] = self.c_[3] * sh_coeffs[0] - self.c_[4] * sh_coeffs[6]
        return M2

    def compute_irradiance(self, normals_aug):
        # evaluate eq. 11 in [1], for one color
        # sh_coeffs in size (9,) for one color
        # assume normals_aug, size (V,4), are normals in camera coordinates

        M = self.compute_irradiance_transfrom(self.sh_coeffs)
        N = normals_aug
        MN = torch.mm(M, N.transpose(0, 1))

        # E = torch.bmm( MN.transpose(0,1).contiguous().view(-1,1,4), N.view(-1,4,1) ).squeeze()
        MN_t = MN.transpose(0, 1).contiguous()
        E = row_wise_vector_product(MN_t, N, dim=0)

        return E

    def forward(self, light, normals):
        # per_vertex_albedo, in size (V,3) or (V*3,)
        # normals, size (V,3), are normals in camera coordinates
        # sh_coeffs, size (27,), coefficients for r, g, b

        # augment the normals

        device = light.device
        v_num = normals.size(0)
        ones = Variable(torch.ones(v_num, 1),
                        requires_grad=False).type(self.dtype_)
        normals_aug = torch.cat((normals, ones), dim=1)

        # compute irradiance
        E_red = self.compute_irradiance(normals_aug, self.sh_coeffs[0:9])
        E_green = self.compute_irradiance(normals_aug, self.sh_coeffs[9:18])
        E_blue = self.compute_irradiance(normals_aug, self.sh_coeffs[18:27])
        E_rgb = torch.cat(
            (E_red.view(-1, 1), E_green.view(-1, 1), E_blue.view(-1, 1)), dim=1)

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
            light = torch.zeros_like(mesh.vertices, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.sh(light, mesh.vertex_normals)
            mesh.textures = mesh.textures * light

        return mesh
