"""

%@author: ntrivisonno

script for bifurcation diagram or logistic map
 
Course: Turbulence in Flux s
Diagrama de bifurcaci'on, Mapa log'istico
Feigenbaum: Boolean value for print the feigenbaum constant
"""

# Importamos paquetes
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

# Flag for the feigenbaum cte
feigenbaum = True
# Flag for ploting random values
plot_values_random = False

def logistic(a, x):
    '''esta función usa ejecuta la 
    función de logística
    Input: x: condición inicial (seria x_n)
           a:parametro de control, escala que gobierna el comportamiento del sistema
    '''
    return a*x*(1-x)

def plot_lin_vert(cte,colores):

    pp = np.linspace(0,1,50)
    oo = cte*np.ones(np.shape(pp))
    plt.plot(oo,pp,color=colores)
    #plt.text(cte+0.05, 0.05, r"a={}".format(cte), color=colores, fontsize=15)

# funcion mapa logistico
plt.figure(figsize = (15,6))
plt.title('Diagrama de Bifurcación')

# Definimos muchos valores de a simultáneamente
na = 10000
a = np.linspace(1, 4.0, na)

# Definamos una condición inicial para cada valor de a
x = 1e-5 * np.ones(na)

n = 1000 #numero de años
# Ejecutemos y grafiquemos x para valores grandes de n (soluciones convergentes)
for i in range(n):
    x = logistic(a, x)
    if i > 950:
        plt.plot(a, x, ',k')

plt.xlabel('$a$')
plt.ylabel('$x$')

# se plotea la cte de Feigenbaum, la cual calcula el pto donde comienza el caos
if feigenbaum:
    cte_feigenbaum = 3.572
    plot_lin_vert(cte_feigenbaum,colores='r')
    plt.text(3.145, 0.95, r"$\delta_{feigenbaum}=%.3f$" %cte_feigenbaum, color="r", fontsize=19)
    plt.text(3.6, 0.0001, r"$\rightarrow CAOS$", color="r", fontsize=19)
    plt.text(2.73, 0.0001, r"$SOL. PERIODICA \leftarrow$", color="b", fontsize=19)

# testing plot line vert
if plot_values_random:
    valores = np.array([1.5, 2.5, 3.2, 3.5])
    colores = ["b","g","c","m","k"]

    #for k in range(valores):
    for k in range(np.shape(valores)[0]):
        plot_lin_vert(valores[k],colores[k])

# ploting logistic map equation
plt.text(1.56 , 0.8, r"$\bf{x_{n+1}=a\;x_n\;(1-x_n)}$", color="k", fontsize=25)

print('#--------------------------------------------')
print('\n FIN, OK!')

plt.show()
