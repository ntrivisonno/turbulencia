# -*- coding: utf-8 -*-
# Script que toma los datos del archivo perfil.txt con las alturas, plotea el perfil y ajusta un perfil logaritmico cercano a la pared
"""
Created on Fri Mar 10 12:45:32 2023

@author: elian & ntrivisonno
"""

#%%Ejercicio 3 ajuste logarítmico
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pp = '/home/zeeburg/Documents/CIMEC/Cursos/turbulencia/actividades/actividad_5/mediciones_ensayo_turbulencias/perfil.txt'

## Lectura de los datos del archivo descargado
df =pd.read_table(pp, sep ='    ',names=['Depth from bottom','Vel'])
df.head()
depth = df['Depth from bottom'] # en cm
vel = df['Vel'] # en cm/s

## Ploteo de los datos
fig1 = plt.figure(figsize = (7,5), dpi=100) # Tamaño y calidad de la figura
plt.plot(df['Vel'],df['Depth from bottom'],'o-b', label = 'Velocidad') # Ploteo del perfil de velocidad
plt.title('Perfil de velocidad', fontsize = 15) # Título del gráfico
plt.xlabel('Vel [cm/s]', fontsize = 12) # Título del eje X
plt.ylabel('Profundidad [cm]', fontsize = 12) # Título del eje Y
plt.ylim(0,35) # Límites del gráfico en X
plt.xlim(7,12) # Límites del gráfico en Y
plt.legend() # Leyenda
plt.grid() # Grilla
plt.show()

## Ploteo semilogarítmico
fig1 = plt.figure(figsize = (7,5), dpi=100) # Tamaño y calidad de la figura
plt.semilogx(depth,vel, 'ob', label = 'Valores observados') # Ploteo del perfil de velocidad
plt.title('Perfil de velocidad', fontsize = 15) # Título del gráfico
plt.ylabel('Vel [cm/s]', fontsize = 12) # Título del eje Y
plt.xlabel('Distancia desde el fondo [cm]', fontsize = 12) # Título del eje X

linear_model = np.polyfit(np.log(depth),vel,1) # Ajuste log al perfil de velocidad
linear_model_fn = np.poly1d(linear_model)      # Coef del polinomio
x_s=np.arange(4,-0.1,-0.01) # Vector de valores de X
# Ploteo semilogarítmico del ajuste
plt.semilogx(np.exp(x_s),linear_model_fn(x_s), label = 'Ajuste logarítmico', color = 'k')
plt.ylim(6,12) # Límites del gráfico en X
plt.xlim(1,40) # Límites del gráfico en Y

plt.legend() # Leyenda
plt.grid() # Grilla
plt.show()

## Iteración para obtener el valor de velocidad de corte
## Physical parameters
kappa_h20 = 0.41
a_r_h20 = 5.29  
nu_h20 = 1 * 10**(-6)

u_star = 1 # Seed vel
LST = 1    # Count error
u = np.zeros(len(vel))
lst = [] 
u_s = [] 
while u_star > 0.0001:
    u_star = u_star - 0.0001
    # velocity update
    u = u_star * a_r_h20 + u_star/kappa_h20 * np.log((depth/100) * u_star / (nu_h20))
    
    LST = (np.sqrt((vel - (u*100))**2)).mean() # Error
    lst.append(LST)
    u_s.append(u_star) 
# Comando que ingresa a la lista de errores y toma el mínimo, luego
# busca su índice y con ese índice busca el u_star que le corresponde
# obteniendo así el u_star asociado al menor error.

u_star_final_wall = u_s[lst.index(min(lst))] # Valor de velocidad de corte final
error_min = min(lst) # Valor de LST mínimo

print(f'- La velocidad de corte en el túnel de viento es de \
{u_star_final_wall:.4f} m^2/s con un mínimo LST de {error_min:.3f}')


'''
#%%Ejercicio 3 ajuste lineal

## Ploteo de los datos
fig1 = plt.figure(figsize = (7,5), dpi=100) # Tamaño y calidad de la figura
plt.plot(df['Vel'],df['Depth from bottom'],'o-b', label = 'Velocidad') # Ploteo del perfil de velocidad
plt.title('Perfil de velocidad', fontsize = 15) # Título del gráfico
plt.xlabel('Velocidad [cm/s]', fontsize = 12) # Título del eje X
plt.ylabel('Profundidad [cm]', fontsize = 12) # Título del eje Y
plt.ylim(0,35) # Límites del gráfico en X
plt.xlim(7,12) # Límites del gráfico en Y

# Ajuste lineal
linear_model = np.polyfit(vel[11:16],depth[11:16]/100,1) # Ajuste lineal al perfil de velocidad
linear_model_fn = np.poly1d(linear_model) # Construcción de la ecuación del ajuste (con cualquier polinomio)
x_s = np.arange(7,10,0.1) # Vector de valores de X

# Ploteo del ajuste lineal
plt.plot(x_s,linear_model_fn(x_s), label = 'Ajuste lineal', color = 'k')
plt.ylim(0,35) # Límites del gráfico en X
plt.xlim(7,12) # Límites del gráfico en Y

plt.legend() # Leyenda
plt.grid() # Grilla
plt.show()

## Iteración para obtener el valor de velocidad de corte
## Parámetros físicos
kappa_agua = 0.41 # Constante de Vön Karman para agua
nu_agua = 0.1 * 10**(-5) # Viscosidad cinemática del agua

u_star = 1 # Valor inicial de la velocidad de corte
MSE = 1 # Valor inicial del error por mínimos cuadrados
u = np.zeros(len(vel[11:16])) # Vector de ceros para el cálculo de la velocidad
mse = [] # Lista vacía para almacenar los valores de MSE de cada iteración
u_s = [] # Lista vacía para almacenar los valores de velocidad de corte de cada iteración
while u_star > 0.0001:
    u_star = u_star - 0.0001 # Variación de la velocidad de corte
    # Cálculo de la velocidad con la velocidad de corte propuesta
    u = u_star * u_star * (depth[11:16]/100) / (nu_agua)
    
    MSE = (np.sqrt((vel[11:16] - u)**2)).mean() # Cálculo del error por mínimos cuadrados
    mse.append(MSE) # Añado valor del error a la lista
    u_s.append(u_star) # Añado valor de la velocidad de corte a la lista
# Comando que ingresa a la lista de errores y toma el mínimo, luego
# busca su índice y con ese índice busca el u_star que le corresponde
# obteniendo así el u_star asociado al menor error.

u_star_final_w = u_s[mse.index(min(mse))] # Valor de velocidad de corte final
error_min = min(mse) # Valor de MSE mínimo

print(f'- La velocidad de corte en el túnel de viento es de \
{u_star_final_w:.4f} cm^2/s con un mínimo MSE de {error_min:.3f}')
'''

print('#--------------------------------------------')
print('\n FIN, OK!')
#plt.show()
