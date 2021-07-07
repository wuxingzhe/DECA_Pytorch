import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def readObj(obj_file, read_tex = False, read_norm = False):
    vertices = []
    texes = []
    normals = []
    faces = []
    r1 = open(obj_file)
    lines = r1.readlines()
    r1.close()

    for line in lines:
        line = line.strip()
        if line[:2] == 'v ':
            params = line.split()
            vertices.append([float(param) for param in params[1:4]])
            if read_tex == True:
                texes.append([float(param) for param in params[4:7]])
        elif line[:2] == 'f ':
            params = line.split()
            faces.append([int((param.split('/'))[0]) for param in params[1:4]])
        elif line[:2] == 'vn':
            params = line.split()
            normals.append([float(param) for param in params[1:4]])

    if read_tex == True:
        if read_norm == True:
            return np.array(vertices), np.array(faces), np.array(texes), np.array(normals)
        else:
            return np.array(vertices), np.array(faces), np.array(texes)
    else:
        if read_norm == True:
            return np.array(vertices), np.array(faces), np.array(normals)
        else:
            return np.array(vertices), np.array(faces)

def readPly(mesh_file):
    vertices = []
    faces = []
    texes = []
    normals = []
    read_data = False;
    read_tex = False; read_norm = False

    r1 = open(mesh_file)
    lines = r1.readlines()
    r1.close()

    pre_lines_len = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line[:8] == 'property':
            params = line.split()
            if params[2] == 'red':
                read_tex = True
                tex_pos = i-pre_lines_len
            elif params[2] == 'x':
                vertex_pos = i-pre_lines_len
            elif params[2] == 'nx':
                read_norm = True
                norm_pos = i-pre_lines_len
        elif line == 'end_header':
            read_data = True
        elif line[:2] == '3 ':
            params = line.split()
            faces.append([int((param.split('/'))[0]) for param in params[1:4]])
        elif read_data == True:
            params = line.split()
            vertices.append([float(param) for param in params[vertex_pos: vertex_pos+3]])
            if read_tex == True:
                texes.append([float(param) for param in params[tex_pos: tex_pos+3]])
            if read_norm == True:
                normals.append([float(param) for param in params[norm_pos: norm_pos+3]])
        else:
            pre_lines_len += 1

    if read_tex == True:
        if read_norm == True:
            return np.array(vertices), np.array(faces), np.array(texes), np.array(normals)
        else:
            return np.array(vertices), np.array(faces), np.array(texes)
    else:
        if read_norm == True:
            return np.array(vertices), np.array(faces), np.array(normals)
        else:
            return np.array(vertices), np.array(faces)
            
# faces index starts from 0
def load_vertices_and_faces_from_Mesh(obj_file):
    vertices = []
    faces = []
    r1 = open(obj_file)
    lines = r1.readlines()
    r1.close()

    if obj_file[-3:] == 'obj':
        for line in lines:
            line = line.strip()
            if line[:2] == 'v ':
                params = line.split()
                vertices.append([float(param) for param in params[1:4]])
            elif line[:2] == 'f ':
                params = line.split()
                faces.append([int(param) for param in params[1:4]])

    elif obj_file[-3:] == 'ply':
        for line in lines:
            line = line.strip()
            if line[:2] == '3 ':
                params = line.split()
                faces.append([int(param) for param in params[1:4]])
            else:
                params = line.split()
                if is_number(params[0]) and is_number(params[1]) and is_number(params[2]):
                    vertices.append([float(param) for param in params[0:3]])

    return np.array(vertices), np.array(faces)

def writeObj(vertices, faces, obj_file, norms = None, texes = None):
    w1 = open(obj_file, 'w')
    for i, vertice in enumerate(vertices):
        w1.write('v '+str(vertice[0])+' '+str(vertice[1])+' '+str(vertice[2]))
        if texes is not None:
            w1.write(' '+str(texes[i,0])+' '+str(texes[i,1])+' '+str(texes[i,2])+'\n')
        else:
            w1.write('\n')
    if norms is not None:
        for norm in norms:
            w1.write('vn '+str(norm[0])+' '+str(norm[1])+' '+str(norm[2])+'\n')
    for face in faces:
        w1.write('f '+str(face[0]+1)+' '+str(face[1]+1)+' '+str(face[2]+1)+'\n')

    w1.close()

def writePly(vertices, faces, mesh_file, normals = None, texes = None):
    w1 = open(mesh_file, 'w')
    w1.write('ply\nformat ascii 1.0\n')
    w1.write('element vertex '+str(vertices.shape[0])+'\n')
    for ss in ['x', 'y', 'z']:
        w1.write('property float '+ss+'\n')
    if normals is not None:
        for ss in ['nx', 'ny', 'nz']:
            w1.write('property float '+ss+'\n')
    if texes is not None:
        for ss in ['red', 'green', 'blue']:
            w1.write('property uchar '+ss+'\n')

    w1.write('element face '+str(faces.shape[0])+'\n')
    w1.write('property list uchar int vertex_indices\n')
    w1.write('end_header\n')

    for i in range(vertices.shape[0]):
        w1.write(str(vertices[i,0])+' '+str(vertices[i,1])+' '+str(vertices[i,2]))
        if normals is not None:
            w1.write(' '+str(normals[i,0])+' '+str(normals[i,1])+' '+str(normals[i,2]))
        if texes is not None:
            w1.write(' '+str(texes[i,0])+' '+str(texes[i,1])+' '+str(texes[i,2]))
        w1.write('\n')

    for face in faces:
        w1.write('3')
        for idx in face:
            w1.write(' '+str(idx))
        w1.write('\n')
    w1.close()
