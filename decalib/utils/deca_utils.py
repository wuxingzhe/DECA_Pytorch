import numpy as np
from .renderer import SRenderY
from . import util
from .rotation_converter import batch_euler2axis

def decompose_code(code, num_dict):
    ''' Convert a flattened parameter vector to a dictionary of parameters
    code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
    '''
    code_dict = {}
    start = 0
    for key in num_dict:
        end = start+int(num_dict[key])
        code_dict[key] = code[:, start:end]
        start = end
        if key == 'light':
            code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
    return code_dict

def displacement2normal(uv_z, coarse_verts, coarse_normals, render):
    ''' Convert displacement map into detail normal map
    '''
    batch_size = uv_z.shape[0]
    uv_coarse_vertices = render.world2uv(coarse_verts).detach()
    uv_coarse_normals = render.world2uv(coarse_normals).detach()
    
    uv_z = uv_z*self.uv_face_eye_mask
    uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
    dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
    uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
    uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
    
    return uv_detail_normals

def displacement2vertex(uv_z, coarse_verts, coarse_normals, render):
    ''' Convert displacement map into detail vertices
    '''
    batch_size = uv_z.shape[0]
    uv_coarse_vertices = render.world2uv(coarse_verts).detach()
    uv_coarse_normals = render.world2uv(coarse_normals).detach()
    
    uv_z = uv_z*self.uv_face_eye_mask
    uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
    dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
    # uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
    # uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
    detail_faces =  self.render.dense_faces

    return dense_vertices, detail_faces
