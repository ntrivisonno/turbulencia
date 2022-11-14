"""

%@author: ntrivisonno

script for bifurcation diagram or logistic map
 
Course: Turbulence in Fluxs
Mapa log'istico: this scripts plots the convergence value of the logistic eq. And has a relation with the difurcation diagram, where the convergence values is different subject to the restriction cte selected.
"""

# Importamos paquetes
import numpy as np
import matplotlib.pyplot as plt

def logistic(a, x):
    '''esta función usa ejecuta la 
    función de logística
    Input: x: condición inicial (seria x_n)
           a:parametro de control, escala que gobierna el comportamiento del sistema
    '''
    return a*x*(1-x)


#--------------------------------------------
# plot
plt.figure(0)
plt.xlabel('$n$')
plt.ylabel('$x$')
a = 3.7 #fijamos a
#a = 3.5
x = 0.01 #fijamos condición inicial
n = 1000

#print('a= ',a)
print('Parámetros físicos seleccionados:\na: {}\nN: {}'.format(a,n))

for i in range(n):
    x = logistic(a, x)
    plt.plot(i, x, 'ok', markersize=1)

plt.title('Solución de la Ec. logísitca, a={}'.format(a))
#plt.text(800, 0.05, r"$x={}$".format(x), fontsize=15)

print('x_añofinal: ', x)

print('#--------------------------------------------')
print('\n FIN, OK!')


plt.show()

