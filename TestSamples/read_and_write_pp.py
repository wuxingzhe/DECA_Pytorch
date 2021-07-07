import numpy as np
import re

# read points from pp file with the sequence 'z,x,y'
def read_from_pp(pp_file):
    r1 = open(pp_file)
    lines = r1.readlines()
    r1.close()

    vertices=[]
    for line in lines:
        line = line.strip()
        if line[:6] == '<point' :
            params = re.split(' |\"', line.strip())
            vertice = []
            for param in params:
                if len(param) <3 or param[0] in ['y', 'x', 'z', '/', 'a', 'n', '=', '<']:
                    continue
                vertice.append(float(param))
            vertices.append([vertice[1], vertice[0],vertice[2]]) # must change the sequence 

    return np.array(vertices)

# write points to pp file with the sequence 'y,z,x'
def write_pp(vertices, pp_file):
    w1 = open(pp_file, 'w')
    w1.write('<!DOCTYPE PickedPoints>\n')
    w1.write('<PickedPoints>\n')
    w1.write(' <DocumentData>\n')
    w1.write(' </DocumentData>\n')

    for i, vertice in enumerate(vertices):
        [x, y, z] = vertice
        w1.write(' <point y="%f" name="%d" active="1" z="%f" x="%f"/>\n' % (y, i, z, x)) # must change the sequence
    w1.write('</PickedPoints>\n')
    w1.close()
