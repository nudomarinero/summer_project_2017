import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def asymetry_analysis():
    data = np.array(pd.read_csv('Workbook2.csv'))
    names, a180_flux, a180_flux_old = data[:,0], data[:,-4], data[:, 1]

    for n in a180_flux:
        print(n)

    key_data = np.array(pd.read_csv('data/key_sel2000.csv'))
    key_names = key_data[:,1]
    key_pmg = key_data[:,2]

    key_pmg_plot = []
    a180_flux_plot = []
    a180_flux_plot_old = []

    for ind,n in enumerate(names):
        try:
            index = np.where(n == key_names)[0][0]
            key_pmg_plot.append(key_pmg[index])
            a180_flux_plot.append(a180_flux[ind])
            a180_flux_plot_old.append(a180_flux_old[ind])
        except:
            pass

    key_pmg_plot = np.array(key_pmg_plot)
    a180_flux_plot = np.array(a180_flux_plot)
    a180_flux_plot_old = np.array(a180_flux_plot_old)

#     print(names[a180_flux_plot>0.97])
#     for a,k in zip(a180_flux_plot, key_pmg_plot):
#         print(a,k)

#     print((key_pmg_plot[key_pmg_plot == 0.]))
    plt.figure()
    plt.plot(key_pmg_plot, a180_flux_plot, '.', markersize=1)
    plt.xlabel('P_MG')
    plt.ylabel('min_a180')

    plt.figure()
    plt.plot(key_pmg_plot, a180_flux_plot_old, '.', markersize=1)
    plt.xlabel('P_MG')
    plt.ylabel('center_a180')


    plt.figure()
    plt.plot(a180_flux_plot, a180_flux_plot_old, '.', markersize=1)
    plt.xlabel('min_a180')
    plt.ylabel('center_a180')

    # plt.figure()
    # plt.plot(a180_flux, a90_flux, '.')
    print('here')
    plt.show()

asymetry_analysis()