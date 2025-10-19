
import sys
import os
import time

# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyfem.materials.material import Material
from pyfem.sections import AreaSection

from pyfem.fromgid import read_gid_data
from pyfem.model_builder import build_model
from pyfem.solvers import solve_linear_static
from pyfem.solvers import solve_linear_static2
from pyfem.togid import to_gid_results

# Materiales
# Para problemas en dinamica el modulo de elasticidad tiene que estar en [N/m2]
steel = Material(elast=1, poiss=0.3, dense=1, 
                 constitutive_model="plane_stress") #[N/m2] [] [Kg/m3]
materials = [steel]

# Secciones
sect1 = AreaSection(thick=0.01) #[m]
sections = [sect1]

def main():
    # 1. Configuración del problema
    input_file = r'D:\Maestria UFRGS\Elementos Finitos\TrabajoFinalEF\ElipseQuad6.gid\ElipseQuad6.dat'
    output_file = r'D:\Maestria UFRGS\Elementos Finitos\TrabajoFinalEF\ElipseQuad6.gid\ElipseQuad6'
    
    # 2. Lectura de datos (Single Responsibility)
    start_time = time.time()
    mesh_data = read_gid_data(input_file)
    time1 = time.time() - start_time
    print(f"Reading data from file  {time1:.3f} seg.")
    
    # 3. Construcción del modelo (Single Responsibility)
    start_time = time.time()
    model = build_model(2, mesh_data, materials, sections)
    time2 = time.time() - start_time
    print(f"Building model          {time2:.3f} seg.")
    
    # 4. Solución del sistema (Single Responsibility)
    start_time = time.time()
    glob_disps, reactions = solve_linear_static2(model)
    node_disps = glob_disps.reshape((model.nnods, model.ndofn))
    time3 = time.time() - start_time
    print(f"Solving system          {time3:.3f} seg.")
    
    # 5. Cálculo de esfuerzos
    start_time = time.time()
    model.calculate_stresses(glob_disps)
    node_cart_stresses, node_prin_stresses, node_vs_stresses = model.calculate_node_stresses()
    time4 = time.time() - start_time
    print(f"Calculating stresses    {time4:.3f} seg.")
    
    # 6. Exportación de resultados (Single Responsibility)
    start_time = time.time()
    to_gid_results(output_file, node_disps, node_cart_stresses, node_prin_stresses, node_vs_stresses)
    time5 = time.time() - start_time
    print(f"Exporting results       {time5:.3f} seg.\n")

    print("Analysis information:")
    print('total exec time:    {:.3f} seg.'.format(time1+time2+time3+time4+time5))
    print('number of nodes:    {0}'.format(model.nnods))
    print('number of DOFs:     {0}'.format(model.ndofs))
    print('number of elements: {0}'.format(len(model.elems)))

    '''
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    #ruta_pickle1 = os.path.join(root, "Trabajo_final_FE", "resultadosT3_4.pkl")
    ruta_pickle2 = os.path.join(root, "Trabajo_final_FE", "resultadosQ4_4.pkl")

    resultados2 = (node_cart_stresses, node_prin_stresses, node_vs_stresses)
    with open(ruta_pickle2, "wb") as f:
    pickle.dump(resultados2, f)
    '''

if __name__ == "__main__":
    main()