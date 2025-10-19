
import numpy as np

def to_gid_results(file_name: str,
                   node_disps: np.ndarray,
                   node_cart_stresses: np.ndarray,
                   node_prin_stresses: np.ndarray,
                   node_vs_stresses: np.ndarray,
                   res_format: str = 'flavia'):
    
    nnods = node_disps.shape[0]

    res_file = f"{file_name}.flavia.res"

    with open(res_file, 'w', encoding='utf8') as fid:
        fid.write("GiD Post Results File 1.0\n")
        fid.write("### \n")

         # Desplazamientos
        fid.write('Result "Displacement" "Load Analysis"  1  Vector OnNodes \n')
        fid.write('ComponentNames "X-Displ", "Y-Displ", "Z-Displ" \n')
        fid.write("Values \n")
        for i, disps in enumerate(node_disps):
            ux, uy = disps
            fid.write(f"{i+1:6d} {ux:13.5e} {uy:13.5e} \n")
        fid.write("End Values \n\n")


        # Tensiones (Matrix OnNodes)
        fid.write('Result "Stress" "Load Analysis"  1  Matrix OnNodes \n')
        fid.write('ComponentNames "Sx", "Sy", "Sz", "Sxy", "Syz", "Sxz" \n')
        fid.write("Values \n")
        if node_cart_stresses.shape[1] == 3:
            for i in range(nnods):
                sx, sy, txy = node_cart_stresses[i,:3]
                fid.write(f"{i+1:6d} {sx:13.5e} {sy:13.5e}  0.0 {txy:13.5e}  0.0  0.0 \n")
        else:
            for i in range(nnods):
                vals = node_cart_stresses[i,:6]
                fid.write(f"{i+1:6d} {vals[0]:13.5e} {vals[1]:13.5e} {vals[2]:13.5e} {vals[3]:13.5e} {vals[4]:13.5e} {vals[5]:13.5e} \n")
        fid.write("End Values \n\n")

        # Tensiones Principales
        fid.write('Result "Principal Stress" "Load Analysis"  1  Matrix OnNodes \n')
        fid.write('ComponentNames "S1", "S2", "S3" \n')
        fid.write("Values \n")

        for i in range(nnods):
            s1, s2, s3 = node_prin_stresses[i,:3]
            fid.write(f"{i+1:6d} {s1:13.5e} {s2:13.5e}  {s3:13.5e} \n")
        fid.write("End Values \n\n")

        # Tension de Von Mises
        fid.write('Result "Von Mises" "Load Analysis"  1  Scalar OnNodes \n')
        fid.write('ComponentNames "VM" \n')
        fid.write("Values \n")
        for i in range(nnods):
            vm = node_vs_stresses[i]
            fid.write(f"{i+1:6d} {vm:13.5e}\n")
        fid.write("End Values \n\n")

    #print(f"Archivo escrito: {res_file}")
    return res_file