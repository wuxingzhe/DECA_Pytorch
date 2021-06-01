import numpy as np
import torch
import math


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    
    return n < 1e-4
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rot_mat_to_euler(R) :
    assert(is_rotation_matrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def angle_to_matrix(theta) :
     
    R_x = np.array([[1,                  0,                   0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0,  math.sin(theta[0]), math.cos(theta[0]) ]])
                           
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0,                  1,                  0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
                 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]),  0],
                    [0,                  0,                   1]])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


def tensor_vertex_transform(mesh, pose):
    s = pose[:, -1].view(-1, 1, 1)
    angles = pose[:, :3]
    R3d = angle_to_matrix(angles).cuda()
    t2d = pose[:, 3:-1].view(-1, 2, 1)
    t3d = torch.zeros(t2d.shape[0], 3)
    t3d[:, :2] = t2d

    transformed_mesh = s * torch.bmm(R3d, mesh) + t3d

    return transformed_mesh


def vertex_transform(vertex, pose):
    s, R, t = pose_to_sRt(pose)
    vertex = s * R.dot(vertex) + t

    return vertex


def pose_to_sRt(pose):
    s = pose[-1].reshape(1, 1)
    angles = pose[:3]
    R = angle_to_matrix(angles)
    t2d = pose[3:-1].reshape(2, 1)
    t3d = np.zeros((3, 1))
    t3d[:2] = t2d

    return s, R, t3d


def save_obj(vertex, load_filename, out_obj_filename):
    with open(out_obj_filename,'w') as out_obj:
        for i in range(int(vertex.shape[0]/3)):
            out_obj.write("v %f %f %f\n" % (vertex[i*3],vertex[i*3+1],vertex[i*3+2]))

        if load_filename is not None:
            with open(load_filename,'r') as obj:
                data = obj.read()
            lines = data.splitlines()
            for line in lines:
                content = line.split()
                if len(content)<1:
                    continue
                if len(content) == 4:
                    if (content[0] == 'vn'):
                        out_obj.write(line+'\n')
                    if (content[0] == 'f'):
                        out_obj.write(line+'\n')
                if len(content) == 3:
                    out_obj.write(f'f {line} \n')

