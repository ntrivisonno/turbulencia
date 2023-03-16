# calc de las frecuencias dominantes
# script for plotting the frequency spectrum of the experimental results in the open channel
'''
@author: ntrivisonno
'''

import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

path ="/home/zeeburg/Documents/CIMEC/Cursos/turbulencia/actividades/actividad_5/otros_datos/Gráficos_de_espectros_de_frecuencias/"

name = "sensc1-"

for i in range(1,17):

    data = np.loadtxt(path+name+str(i)+".FFT", delimiter=';', skiprows=11)
    #"Frequency";"FFT-V Magnitude_0";""

    f = data[:,0]
    fft = data[:,1]

    plt.figure()
    plt.loglog(f,fft,'-')
    plt.ylim(0.0, 0.1)
    #plt.ylim(0.0, 1)
    #plt.title("Case sensc1-",i)

print('#--------------------------------------------')
print('\n FIN, OK!')
plt.show()
