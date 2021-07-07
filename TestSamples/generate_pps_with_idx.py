import os, glob
import numpy as np
from read_and_write_obj import readObj
from read_and_write_pp import write_pp

pps_idx = np.load('FLAME_mesh_pp_idx.npy')

src_path = 'vggface2/results15'
for id_name in os.listdir(src_path):
    id_name = id_name.strip()
    mesh_path = os.path.join(src_path, id_name)
    if not os.path.isdir(mesh_path):
        continue
    vertices, faces = readObj(os.path.join(mesh_path, id_name+'_albedo.obj'))
    write_pp(vertices[pps_idx, :], os.path.join(mesh_path, id_name+'_albedo.pp'))
