
import numpy as np

def read_gid_data(file_path):
    nodes = []
    elements = []
    restraints = []
    nodeloads = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    part_coordinates = False
    part_elements = False
    part_restraints = False
    part_nod_loads = False

    for line in lines:
        line = line.strip()

        if line.startswith('# Coordinates'):
            part_coordinates = True
            continue
        elif line.startswith('# End Coordinates'):
            part_coordinates = False
            continue
        elif line.startswith('# Elements'):
            part_elements = True
            continue
        elif line.startswith('# End Elements'):
            part_elements = False
            continue 
        elif line.startswith('# Restraints'):
            part_restraints = True
            continue
        elif line.startswith('# End Restraints'):
            part_restraints = False
            continue
        elif line.startswith('# Nodal Loads'):
            part_nod_loads = True
            continue
        elif line.startswith('# End Nodal Loads'):
            part_nod_loads = False
            continue

        if part_coordinates:
            data = line.split()
            nodes.append([float(data[1]), float(data[2])])

        elif part_elements:
            data = line.split()
            #tag = int(data[0])-1
            etype = int(data[1])
                
            if etype == 3:  # Quad4 en GiD
                elements.append([
                    int(1),
                    'Quad4',
                    int(data[2])-1, #material/section id (el usuario nombra en GiD)
                    # orden horario formato Nastran
                    [int(data[6])-1,  # node4
                     int(data[5])-1,  # node3
                     int(data[4])-1,  # node2
                     int(data[3])-1]  # node1 
                ])
            elif etype == 2:  # Tri3 en GiD
                elements.append([
                    int(1),
                    'Tri3',
                    int(data[2])-1, #material/section id (el usuario nombra en GiD)
                    # orden horario formato Nastran
                    [int(data[5])-1,  # node 3
                     int(data[4])-1,  # node 2
                     int(data[3])-1]  # node 1
                ])

        elif part_restraints:
            data = line.split()
            # list: [node_tag, restraint_x, restraint_y]
            restraints.append([int(data[0])-1, [int(data[1]), int(data[2])]])

        elif part_nod_loads:
            data = line.split()
            # list: [node_tag, load_x, loadt_y]
            nodeloads.append([int(data[0])-1, [float(data[1]), float(data[2])]])

    return {
        'nodes': np.array(nodes),
        'elements': elements,
        'restraints': restraints,
        'nodeloads': nodeloads
        }
