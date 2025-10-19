
from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import FrameSection, AreaSection


def build_model(ndofn, mesh_data, materials, sections):

    mod = Model(ndofn)
    mod.add_materials(materials)
    mod.add_sections(sections)
    mod.add_nodes(mesh_data['nodes'])


    for elem_data in mesh_data['elements']:
        if elem_data[0] == 0:
             mod.add_frame_element(
                  etype = elem_data[1],
                  material = materials[elem_data[2]],
                  section = sections[elem_data[2]],
                  conec = elem_data[3],    
             )
        elif elem_data[0] == 1:
             mod.add_area_element(
                  etype = elem_data[1],
                  material = materials[elem_data[2]],
                  section = sections[elem_data[2]],
                  conec = elem_data[3]
             )
        elif elem_data[0] == 2:
             mod.add_solid_element(
                  etype = elem_data[1],
                  material = materials[elem_data[2]],
                  conec = elem_data[3]
             )


    for restraint_data in mesh_data['restraints']:
            mod.add_node_restraint(
                restraint_data[0],
                restraint_data[1]
            )
        
        # Agregar cargas
    for load_data in mesh_data['nodeloads']:
        mod.add_node_load(
            load_data[0],
            load_data[1]
        )
        
    return mod


    
