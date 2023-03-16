# Script for post-processing experimental values - flowchanel, openchannel
# Optimization code of main.py
'''
@author: ntrivisonno
'''
import numpy as np
import matplotlib.pyplot as plt

eps = np.finfo(float).eps
l0 = 34.3  # height of the sensor [cm]
nu = 1.004  # cinematic viscosity [mm2/s]

mediciones = [f'sensc1-{i}.Vu' for i in range(1, 3)]

for medicion in mediciones:
    data = np.loadtxt(medicion, delimiter=';', skiprows=9)

    t, u, v, w = data[:, 0], data[:, 3], data[:, 4], data[:, 5]

    #mean_vec = np.zeros((np.shape(t)[0], 1))
    mean_vec = np.zeros((np.shape(t)[0]))
    # means as vector
    mean_vec[:] = np.mean([u, v, w], axis=0)
    u_mean_vec, v_mean_vec, w_mean_vec = mean_vec, mean_vec, mean_vec

    # fluctuations
    u_fl, v_fl, w_fl = u - u_mean_vec.flatten(), v - v_mean_vec.flatten(), w - w_mean_vec.flatten()

    # Fluctations velocity products
    uu, uv, uw, vv, vw, ww = u_fl * v_fl, u_fl * v_fl, u_fl * w_fl, v_fl * v_fl, v_fl * w_fl, w_fl * w_fl

    uu_mean, uv_mean, uw_mean = np.mean(uu), np.mean(uv), np.mean(uw)
    vv_mean, vw_mean, ww_mean = np.mean(vv), np.mean(vw), np.mean(ww)

    tensor_reynolds = np.array([[uu_mean, uv_mean, uw_mean], [uv_mean, vv_mean, vw_mean], [uw_mean, vw_mean, ww_mean]])

    # TKE - Turbulent Kinetic Energy
    tke = 0.5 * np.trace(tensor_reynolds)

    # Relation between Large & Kolmogorov scale
    u_mean = np.mean([u, v, w])
    Re = u_mean * 10 * (l0 * 10) / nu  # Reynolds number

    # Kolmogorov
    eta = (l0 * 10) * (Re ** (-3 / 4))
    u_eta = (u_mean * 10) * (Re ** (-1 / 4))
    tau_eta = eta / u_eta

    # Kolmogorov's Reynolds - Verification
    Re_k = eta * u_eta / nu

    # Taylor
    lambda1 = eta * (Re ** (1 / 4))
    lambda2 = (l0 * 10) * Re ** (-1 / 2)

    # Create subplots and plot all the graphs
    fig, axs = plt.subplots(2, figsize=(7, 8))
    axs = axs.ravel()

    axs[0].plot(t, u, label='Vel X')
    axs[0].plot(t, v, label='Vel Y')
    axs[0].plot(t, w, label='Vel Z')
    axs[0].plot(t, u_mean_vec, label='$\overline{u}$')
    axs[0].plot(t, v_mean_vec, label='$\overline{v}$')
    axs[0].plot(t, w_mean_vec, label='$\overline{w}$')
    plt.title("Velocidades $u=\overline{u}+u'$")
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Vel [cm/s]')
    axs[0].grid()
    axs[1].plot(u,w,'.')
    axs[1].set_title('Component uw')
    axs[1].set_xlabel('u [cm/s]')
    axs[1].set_ylabel('w [cm/s]')
    axs[1].grid()
    fig.legend()

    print('#--------------------------------------------')
    print("Verifications:")
    print("Re_k: {}\nlambda1: {}\nlambda2: {}".format(Re_k,lambda1,lambda2))

print('#--------------------------------------------')
print('\n FIN, OK!')
plt.show()
