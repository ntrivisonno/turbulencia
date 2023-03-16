# Script for post-processing experimental values - flowchanel, openchannel
'''
@author: ntrivisonno
'''

import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

# Upload all the cases and plot
for i in range(1,3):
    medicion = i
    medicion = 'sensc1-'+str(medicion)+'.Vu'
    data = np.loadtxt(medicion, delimiter=';', skiprows=9)
    #"Time, seconds";"Position";"Flag";"Vx_0";"Vy_0";"Vz_0";"COR0_0";"COR1_0";"COR2_0";"SNR0_0";"SNR1_0";"SNR2_0";"AMP0_0";"AMP1_0";"AMP2_0";"CORAvg_0";"SNRAvg_0";"AmpAvg_0"

    t = data[:,0]
    u = data[:,3] #vx
    v = data[:,4] #vy
    w = data[:,5] #vz

    # physical parameters
    nu = 1.004 # cinematic viscosity [mm2/s]
    l0 = 34.3 # higth of the sensor [cm]

    # Stadistics
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    w_mean = np.mean(w)
    # means as vector
    u_mean_vec = np.zeros((np.shape(t)[0],1))
    v_mean_vec = np.zeros((np.shape(t)[0],1))
    w_mean_vec = np.zeros((np.shape(t)[0],1))
    # fluctuations
    u_fl = np.zeros((np.shape(t)[0],1))
    v_fl = np.zeros((np.shape(t)[0],1))
    w_fl = np.zeros((np.shape(t)[0],1))
    for i in range(np.shape(t)[0]):
        u_mean_vec[i] = u_mean
        v_mean_vec[i] = v_mean
        w_mean_vec[i] = w_mean
        u_fl[i] = u[i] - u_mean
        v_fl[i] = v[i] - v_mean
        w_fl[i] = w[i] - w_mean

    # Fluctations velocity products
    uu = u_fl*v_fl
    uv = u_fl*v_fl
    uw = u_fl*w_fl
    vv = v_fl*v_fl
    vw = v_fl*w_fl
    ww = w_fl*w_fl


    uu_mean = np.mean(uu)
    uv_mean = np.mean(uv)
    uw_mean = np.mean(uw)
    vv_mean = np.mean(vv)
    vw_mean = np.mean(vw)
    ww_mean = np.mean(ww)

    tensor_reynolds = np.array([[uu_mean, uv_mean, uw_mean],
                                [uv_mean, vv_mean, vw_mean],
                                [uw_mean, vw_mean, ww_mean]])
    
    # TKE - Turbulent Kinetic Energy
    tke = 0.5*np.trace(tensor_reynolds)

    # Relation between Large & Kolmogorov scale
    Re = (u_mean*10)*(l0*10)/nu # Reynolds number

    # Kolmogorov
    eta = (l0*10)*(Re**(-3/4))
    u_eta = (u_mean*10)*(Re**(-1/4))
    tau_eta = eta/u_eta

    # Kolmogorov's Reynolds - Verification
    Re_k =  eta*u_eta/nu

    # Taylor
    lambda1 = eta*(Re**(1/4))
    lambda2 = (l0*10)*Re**(-1/2)

    plt.figure()
    plt.plot(t,u, label='Vel X')
    plt.plot(t,v, label='Vel Y')
    plt.plot(t,w, label='Vel Z')
    plt.plot(t,u_mean_vec, label='$\overline{u}$')
    plt.plot(t,v_mean_vec, label='$\overline{v}$')
    plt.plot(t,w_mean_vec, label='$\overline{w}$')
    plt.title("Velocidades $u=\overline{u}+u'$")
    plt.xlabel('Time [s]')
    plt.ylabel('Vel [cm/s]')
    plt.legend()

    # Reynolds stress plot
    fig_size = (8,7)
    f,ax = plt.subplots(2,figsize=fig_size)
    f.suptitle('Reynolds Stress Tensor')
    ax[0].plot(u,v,'.')
    ax[0].set_title('Component uv')
    #ax[0].set_xlabel('u [cm/s]')
    ax[0].set_ylabel('v [cm/s]')
    ax[0].grid()
    ax[1].plot(u,w,'.')
    ax[1].set_title('Component uw')
    ax[1].set_xlabel('u [cm/s]')
    ax[1].set_ylabel('w [cm/s]')
    ax[1].grid()


    # Console out
    print("Sensor: {}".format(medicion))
    # header overleaf
    #print("medicion, u_mean, v_mean, w_mean, tke")
    print("{} & {} & {} & {} & {} \\ ".format(medicion, u_mean, v_mean, w_mean, tke))


    #print("medicion, tensor_reynolds[0,0], tensor_reynolds[0,1], tensor_reynolds[0,2], tensor_reynolds[1,1], tensor_reynolds[1,2], tensor_reynolds[2,2]")
    print("{} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\ ".format(medicion, tensor_reynolds[0,0], tensor_reynolds[0,1], tensor_reynolds[0,2], tensor_reynolds[1,1], tensor_reynolds[1,2], tensor_reynolds[2,2]))

    #print("medicion, eta, u_eta, tau_eta")
    print("{} & {} & {} & {} \\".format(medicion, eta, u_eta, tau_eta))
    
    #print("medicion, Re_k, lambda1, lambda2")
    print("{} & {} & {} & {} \\".format(medicion, Re_k, lambda1, lambda2))

    print('#--------------------------------------------')
    print("Verifications:")
    print("Re_k: {}\nlambda1: {}\nlambda2: {}".format(Re_k,lambda1,lambda2))
    print('#--------------------------------------------')
    print('\n FIN, OK!')
plt.show()

