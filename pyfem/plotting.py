
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tabulate import tabulate
from colorama import Fore, Style, init

init(autoreset=True)
#'''
def print_matrix(array, width, floatfmt=".2f"):
    df = pd.DataFrame(array)
    
    # Formatear filas manualmente
    rows = []
    for idx, row in df.iterrows():
        formatted_row = [f"{Fore.RED}{idx:{width}}{Style.RESET_ALL}"] + \
                        [f"{x:{width}{floatfmt[1:]}}" for x in row]
        rows.append(formatted_row)

    # Formatear headers
    headers = [""] + [f"{Fore.RED}{col:{width}}{Style.RESET_ALL}" for col in df.columns]

    # Generar tabla
    new_table = tabulate(rows, headers=headers, tablefmt="plain", floatfmt=floatfmt)   
    print(new_table)

#'''
'''
def print_matrix(array, width, floatfmt=".2f"):
    df = pd.DataFrame(array)
    df_formatted = df.map(lambda x: f"{x:{width}}")
    df_formatted.index = [f"{Fore.RED}{idx:{width}}{Style.RESET_ALL}" for idx in df.index]
    headers = [f"{Fore.RED}{col:{width}}{Style.RESET_ALL}" for col in df.columns]
    new_table = tabulate(df_formatted, headers=headers, showindex=True, tablefmt="plain", floatfmt=floatfmt)
    print(new_table)
'''

def plot_matrix(Mat, part=1):
    figure = plt.figure()
    plt.spy(Mat, origin='upper', extent=(0, Mat.shape[1], Mat.shape[0], 0))
    num_rows, num_cols = Mat.shape
    plt.xticks(np.arange(0, num_cols + 1, part))  # Ticks enteros para las columnas
    plt.yticks(np.arange(0, num_rows + 1, part))  # Ticks enteros para las filas
    plt.grid(True)
    return figure

def plot_imshow(Mat, n, part=1):
    figure = plt.figure(n)
    plt.imshow(Mat, cmap='viridis', origin='upper', 
           interpolation='none',  # Evitar interpolación
           extent=(0, Mat.shape[1], Mat.shape[0], 0))
    plt.colorbar()
    num_rows, num_cols = Mat.shape
    plt.xticks(np.arange(0, num_cols + 1, part))  # Ticks enteros para las columnas
    plt.yticks(np.arange(0, num_rows + 1, part))  # Ticks enteros para las filas
    plt.grid(True)
    return figure



def plot_2dmodel(mod,):

    plt.figure()
    ax = plt.gca()

    for elem in mod.elems:
        x = elem.coord[:,0]
        y = elem.coord[:,1]
        ax.plot(x, y, 'b-o', lw=1, markersize=2)

    for n, coor in enumerate(mod.coord):
        ax.text(coor[0]+0.05, coor[1]+0.05, str(n), 
                fontsize=7, ha='left', va='bottom')
        
    ax.axis('equal')

    return ax


def plot_3dmodel(mod):

    x_lines, y_lines, z_lines = [], [], []
    
    for elem in mod.elems:
        node1, node2 = elem.coord

        x_lines.extend([node1[0], node2[0], None])
        y_lines.extend([node1[1], node2[1], None])
        z_lines.extend([node1[2], node2[2], None])

    x_nodes = mod.coord[:, 0]
    y_nodes = mod.coord[:, 1]
    z_nodes = mod.coord[:, 2]

    fig = go.Figure()
    # Añadir barras (líneas)
    fig.add_trace(go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color='blue', width=5),
        name='Truss'))
    
    '''
    fig.add_trace(go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color='green', dash='dash', width=3),
        name='Truss'))
    '''
    
    # Añadir nodos
    fig.add_trace(go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Nodes'))

    # Ajustar aspecto
    fig.update_layout(
        title='Truss 3D display',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )

    return fig