
import numpy as np
import matplotlib.pyplot as plt

# Parámetros de entrada
q0 = 7000 #7000
e = 2.08e10 #2.08e10
nu = 0.3 #0.3
t = 0.1
a = 4
b = 4
nte = 9

# Cálculo de la rigidez de flexión
d = e * t**3 / (12 * (1 - nu**2))

# Crear malla de puntos
x, y = np.meshgrid(np.arange(0, a, 0.1), np.arange(0, b, 0.1))

# Inicializar matrices para los resultados
w1 = np.zeros_like(x)
mx = np.zeros_like(x)
my = np.zeros_like(x)
mxy = np.zeros_like(x)

# Calcular series de Navier
for m in range(1, nte + 1, 2):
    for n in range(1, nte + 1, 2):
        term = 1 / (m * n * (n**2 * a**2 + m**2 * b**2)**2)
        w1 += term * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)
        
        term_mx = (b**2 * m**2 + nu * a**2 * n**2) / (m * n * (n**2 * a**2 + m**2 * b**2)**2)
        mx += term_mx * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)
        
        term_my = (a**2 * n**2 + nu * b**2 * m**2) / (m * n * (n**2 * a**2 + m**2 * b**2)**2)
        my += term_my * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)
        
        term_mxy = 1 / (n**2 * a**2 + m**2 * b**2)**2
        mxy += term_mxy * np.cos(m * np.pi * x / a) * np.cos(n * np.pi * y / b)

# Aplicar factores de escala
w1 *= 16 * q0 * a**4 * b**4 / (np.pi**6 * d)
mx *= 16 * q0 * a**2 * b**2 / np.pi**4
my *= 16 * q0 * a**2 * b**2 / np.pi**4
mxy *= -(1 - nu) * 16 * q0 * a**3 * b**3 / np.pi**2

print('Max. displacement: {:5f} [m]'.format(np.max(w1)))

# Graficar resultados
plt.figure(1)
plt.contourf(x, y, w1)
plt.title('Deslocamento W [m]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.colorbar()

plt.figure(2)
plt.contourf(x, y, mx)
plt.title('Momento mx [Nm/m]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.colorbar()

plt.figure(3)
plt.contourf(x, y, my)
plt.title('Momento my [Nm/m]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.colorbar()

plt.figure(4)
plt.contourf(x, y, mxy)
plt.title('Momento mxy [Nm/m]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.colorbar()

plt.show()